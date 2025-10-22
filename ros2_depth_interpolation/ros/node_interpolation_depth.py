from collections import deque
import math
import threading
import time

import numpy as np
import torch

from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Cache, Subscriber as SubscriberFilter
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, ReliabilityPolicy, QoSProfile
import rclpy.qos as qos
from rcl_interfaces.msg import FloatingPointRange, IntegerRange, ParameterDescriptor, ParameterType
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from nimbro_interfaces.srv import ProjectDome, ColorizePoints
import nimbro_utils.compat.point_cloud2 as point_cloud2
from nimbro_utils.parameter_handler import ParameterHandler
from nimbro_utils.tf_oracle import TFOracle

from ros2_depth_interpolation.projection import ModelDoubleSphere, ModelRationalPolynomial
from ros2_depth_interpolation.sampling import SamplerColor, SamplerDepth


class NodeInterpolationDepth(Node):
    info_rational_polynomial_default = dict(
        height=720,
        width=1280,
        fx=689.3404541015625,
        fy=689.5661010742188,
        cx=642.3411865234375,
        cy=352.51873779296875,
    )

    info_double_sphere_default = dict(
        height=1920,
        width=2556,
        fx=571.4448814063372,
        fy=570.6090733170088,
        cx=1334.1674886529015,
        cy=985.4219057464759,
        xi=-0.2996439614293713,
        alpha=0.5537226081641069,
    )

    def __init__(
        self,
        name_frame_camera="camera_ids_link",
        name_frame_lidar="os_sensor_link",
        slop_synchronizer=0.5,
        topic_image="/camera_ids/image_color",
        topic_image_rectified="/camera_ids/rectified/image_color",
        topic_info="/camera_ids/camera_info",
        topic_info_rectified="/camera_ids/rectified/camera_info",
        topic_projected_depth="/camera_ids/projected/depth/image",
        topic_projected_points="/ouster/projected/points",
        topic_points="/ouster/points",
        color_invalid="(255, 87, 51)",
        factor_downsampling=8,
        use_color_sampling=True,
        use_depth_sampling=True,
        use_rectification=False,
        use_service_only=False,
        use_knn_interpolation=True,
        k_knn=1,
        mode_interpolation="nearest",
        size_cache=10,
    ):
        super().__init__(node_name="interpolation_depth")

        self.bridge_cv = None
        self.cache_times_points_message = None
        self.color_invalid = color_invalid
        self.coords_uv_full_flat = None
        self.device = None
        self.handler_parameters = None
        self.k_knn = k_knn
        self.lock = None
        self.mode_interpolation = mode_interpolation
        self.name_frame_camera = name_frame_camera
        self.name_frame_lidar = name_frame_lidar
        self.publisher_depth = None
        self.publisher_image_rectified = None
        self.publisher_info_rectified = None
        self.publisher_points = None
        self.profile_qos = None
        self.factor_downsampling = factor_downsampling
        self.sampler_color = None
        self.sampler_depth = None
        self.service_colorize_points = None
        self.service_project_dome = None
        self.size_cache = size_cache
        self.slop_synchronizer = slop_synchronizer
        self.subscriber_image = None
        self.subscriber_info = None
        self.subscriber_points = None
        self.tf_broadcaster = None
        self.tf_buffer = None
        self.tf_listener = None
        self.tf_oracle = None
        self.topic_image = topic_image
        self.topic_image_rectified = topic_image_rectified
        self.topic_info = topic_info
        self.topic_info_rectified = topic_info_rectified
        self.topic_projected_depth = topic_projected_depth
        self.topic_projected_points = topic_projected_points
        self.topic_points = topic_points
        self.use_color_sampling = use_color_sampling
        self.use_depth_sampling = use_depth_sampling
        self.use_rectification = use_rectification
        self.use_service_only = use_service_only
        self.use_knn_interpolation = use_knn_interpolation

        self._init()

    def _init(self):
        self.cache_times_points_message = deque([], maxlen=self.size_cache)
        self._init_projector()

        self.bridge_cv = CvBridge()
        self._init_tf_oracle()

        self.handler_parameters = ParameterHandler(self, verbose=False)
        self._init_parameters()

        self._del_publishers()
        self._init_publishers()
        self._del_services()
        self._init_services()
        self._del_subscribers()
        self._init_subscribers()

    def _init_tf_oracle(self):
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_oracle = TFOracle(self)

    def _init_projector(self):
        # TODO
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lock = threading.Lock()

        self.sampler_depth = SamplerDepth(
            factor_downsampling=self.factor_downsampling,
            k_knn=self.k_knn,
            mode_interpolation=self.mode_interpolation,
        )
        self.sampler_color = SamplerColor(color_invalid=self.color_invalid)

    def _init_publishers(self):
        profile_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)

        self.publisher_depth = self.create_publisher(
            msg_type=Image,
            topic=self.topic_projected_depth,
            qos_profile=qos.qos_profile_sensor_data,
            callback_group=ReentrantCallbackGroup(),
        )
        self.publisher_points = self.create_publisher(
            msg_type=PointCloud2,
            topic=self.topic_projected_points,
            qos_profile=qos.qos_profile_sensor_data,
            callback_group=ReentrantCallbackGroup(),
        )
        self.publisher_image_rectified = self.create_publisher(
            msg_type=Image,
            topic=self.topic_image_rectified,
            qos_profile=qos.qos_profile_sensor_data,
            callback_group=ReentrantCallbackGroup(),
        )
        self.publisher_info_rectified = self.create_publisher(
            msg_type=CameraInfo,
            topic=self.topic_image_rectified,
            qos_profile=qos.qos_profile_sensor_data,
            callback_group=ReentrantCallbackGroup(),
        )

    def _del_publishers(self):
        names_publisher = ["publisher_depth", "publisher_points", "publisher_image_rectified", "publisher_info_rectified"]
        for name_publisher in names_publisher:
            publisher = getattr(self, name_publisher)
            if publisher is not None:
                self.destroy_publisher(publisher)
                setattr(self, name_publisher, None)

    def _init_subscribers(self):
        self.subscriber_info = SubscriberFilter(
            self,
            CameraInfo,
            self.topic_info,
            qos_profile=qos.qos_profile_sensor_data,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.cache_info = Cache(self.subscriber_info, self.size_cache)

        if self.use_service_only:
            return

        # ApproximateTimeSynchronizer not working as expected. Slop is disregarded and messages are often reused more than once
        # self.synchronizer = ApproximateTimeSynchronizer(fs=[self.subscriber_points, self.subscriber_image, self.subscriber_info], queue_size=3, slop=self.slop_synchronizer)
        # self.synchronizer.registerCallback(self.on_messages_received_callback)

        if self.use_depth_sampling or self.use_color_sampling:
            self.subscriber_points = SubscriberFilter(
                self,
                PointCloud2,
                self.topic_points,
                qos_profile=qos.qos_profile_sensor_data,
                callback_group=MutuallyExclusiveCallbackGroup(),
            )
            self.cache_points = Cache(self.subscriber_points, self.size_cache)
            self.cache_points.registerCallback(self.on_message_points_received_callback)

        if self.use_color_sampling or self.use_rectification:
            self.subscriber_image = SubscriberFilter(
                self,
                Image,
                self.topic_image,
                qos_profile=qos.qos_profile_sensor_data,
                callback_group=MutuallyExclusiveCallbackGroup(),
            )
            self.cache_image = Cache(self.subscriber_image, self.size_cache)
            self.cache_image.registerCallback(self.on_message_image_received_callback)

    def _del_subscribers(self):
        if self.subscriber_info is not None:
            self.destroy_subscription(self.subscriber_info.sub)
            self.cache_info = None
            self.subscriber_info = None

        if self.subscriber_points is not None:
            self.destroy_subscription(self.subscriber_points.sub)
            self.cache_points = None
            self.subscriber_points = None

        if self.subscriber_image is not None:
            self.destroy_subscription(self.subscriber_image.sub)
            self.cache_image = None
            self.subscriber_image = None

    def _init_services(self):
        self.service_project_dome = self.create_service(
            ProjectDome,
            "/interpolation_depth/project_dome",
            self.on_service_call_project_dome,
            callback_group=ReentrantCallbackGroup(),
        )
        self.service_colorize_points = self.create_service(
            ColorizePoints,
            "/interpolation_depth/colorize_points",
            self.on_service_call_colorize_points,
            callback_group=ReentrantCallbackGroup(),
        )

    def _del_services(self):
        names_service = ["service_project_dome", "service_colorize_points"]
        for name_service in names_service:
            service = name_service
            if service is not None:
                self.destroy_service(service)
                setattr(self, name_service, None)

    def publish_image_depth(self, image, name_frame, stamp):
        image_msg = self.bridge_cv.cv2_to_imgmsg(
            image,
            header=Header(frame_id=name_frame, stamp=stamp),
            encoding="mono16",
        )
        self.publisher_depth.publish(image_msg)

    def publish_points(self, message_pointcloud, pointcloud_colored, offset):
        points_msg = point_cloud2.create_cloud(
            header=message_pointcloud.header,
            fields=message_pointcloud.fields + [PointField(name="rgb", offset=offset, datatype=PointField.UINT32, count=1)],
            points=pointcloud_colored,
        )
        self.publisher_points.publish(points_msg)

    def publish_image_rectified(self, image, name_frame, stamp):
        image_msg = self.bridge_cv.cv2_to_imgmsg(
            image,
            header=Header(frame_id=name_frame, stamp=stamp),
            encoding="rgb8",
        )
        self.publisher_image_rectified.publish(image_msg)

    def publish_info_rectified(self, info, name_frame, stamp):
        info_msg = CameraInfo(
            **info,
            header=Header(frame_id=name_frame, stamp=stamp),
        )

        self.publisher_info_rectified.publish(info_msg)

    def points2tensor(self, pointcloud):
        """Return image as tensor of shape [B, C, HxW]"""
        points = pointcloud[["x", "y", "z"]]
        points = np.lib.recfunctions.structured_to_unstructured(points)
        points = torch.as_tensor(points, dtype=torch.float16, device=self.device)
        points = points.permute(1, 0)
        points = points[None, ...]

        return points

    def image2tensor(self, image):
        """Return image as tensor of shape [B, C, H, W]"""
        image = torch.as_tensor(image, device=self.device)
        image = image.permute(2, 0, 1)
        images = image[None, ...]
        return images

    def colors_to_numpy(self, colors):
        colors = colors.detach().cpu().numpy().astype(np.uint32)

        r = colors[0]
        g = colors[1]
        b = colors[2]
        colors = (r << 16) | (g << 8) | b

        return colors

    def create_dtype_with_rgb(self, dtype):
        # There is no better way to access this >:(
        # Also np.lib.recfunctions.append_fields does not work with offsets, formats, itemsize
        formats = [field[0] for field in dtype.fields.values()]
        offsets = [field[1] for field in dtype.fields.values()]
        offset = offsets[-1] + np.dtype(formats[-1]).itemsize

        dtype = np.dtype(
            {
                "names": list(dtype.names) + ["rgb"],
                "formats": formats + ["<u4"],
                "offsets": offsets + [offset],
                # Round up to next multiple of 8
                "itemsize": int(math.ceil((offset + 4) / 8)) * 8,
            }
        )

        return dtype, offset

    @torch.inference_mode()
    def rectify_images(self, images, model_camera, model_camera_original):
        x = torch.linspace(-model_camera.shape_image[2] / 2, model_camera.shape_image[2] / 2 - 1, model_camera.shape_image[2], device=self.device)
        y = torch.linspace(-model_camera.shape_image[1] / 2, model_camera.shape_image[1] / 2 - 1, model_camera.shape_image[1], device=self.device)
        z = math.sqrt(model_camera.fx * model_camera.fy)

        y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
        point3D = torch.stack([x_grid, y_grid, torch.full_like(x_grid, z)], dim=0)
        point3D = point3D.reshape((3, -1))[None, ...]
        point3D /= torch.norm(point3D, dim=1, keepdim=True) + 1e-8

        coords_uv, mask_valid = model_camera_original.project_points_onto_image(point3D)
        coords_uv[:, 0, :] = 2.0 * coords_uv[:, 0, :] / model_camera.shape_image[2] - 1.0
        coords_uv[:, 1, :] = 2.0 * coords_uv[:, 1, :] / model_camera.shape_image[1] - 1.0
        coords_uv = coords_uv.permute(0, 2, 1)

        # grid_sample not implemented for dtype byte
        images = images.float() / 255.0
        output = torch.nn.functional.grid_sample(input=images, grid=coords_uv.float()[:, None, :, :], align_corners=True, mode="bilinear")

        output[:, 0, :].masked_fill_(~mask_valid, 0)
        output[:, 1, :].masked_fill_(~mask_valid, 0)
        output[:, 2, :].masked_fill_(~mask_valid, 0)

        output = output.reshape((1, 3, model_camera.shape_image[1], model_camera.shape_image[2]))
        output = (output * 255.0).byte()

        return output

    def compute_pointcloud_colored(self, coords_uv, images, pointcloud, mask_valid=None):
        colors = self.sampler_color(coords_uv, images, mask_valid, use_half_precision=True)
        colors = colors[0]
        colors = self.colors_to_numpy(colors)

        dtype, offset = self.create_dtype_with_rgb(pointcloud.dtype)
        # Filling a new array for performance (see: https://stackoverflow.com/questions/25427197/numpy-how-to-add-a-column-to-an-existing-structured-array)
        pointcloud_colored = np.empty(pointcloud.shape, dtype=dtype)
        names_fields_before = list(pointcloud.dtype.names)
        pointcloud_colored[names_fields_before] = pointcloud[names_fields_before]
        pointcloud_colored["rgb"] = colors

        return pointcloud_colored, offset

    def compute_depth_image(self, coords_uv, points, mask_valid=None):
        image_depth = self.sampler_depth(coords_uv, points, mask_valid, use_knn_interpolation=self.use_knn_interpolation)
        image_depth = image_depth[0]
        image_depth = image_depth.permute(1, 2, 0)

        image_depth = image_depth.detach().cpu().numpy().astype(np.uint16)

        return image_depth

    # Apparently, messages are received in correct order already (based on a few inspected samples), so this is not necessary
    def get_newest_element_before_time(self, cache, time_before):
        """Custom function to replace cache.getElemBeforeTime which does not order"""
        message_newest_before = None
        time_newest_before = None
        for message, time in zip(cache.cache_msgs, cache.cache_times):
            if time <= time_before and (time_newest_before is None or time_newest_before < time):
                message_newest_before = message
                time_newest_before = time

        return message_newest_before

    def on_message_points_received_callback(self, message_points):
        time_message = Time.from_msg(message_points.header.stamp)
        duration_latency = (self.get_clock().now() - time_message).nanoseconds / 1e9
        self.get_logger().info(f"Received points message with latency {duration_latency}", throttle_duration_sec=1.0)
        self.on_messages_received_callback(time_message, message_points=message_points)

    def on_message_image_received_callback(self, message_image):
        time_message = Time.from_msg(message_image.header.stamp)
        duration_latency = (self.get_clock().now() - time_message).nanoseconds / 1e9
        self.get_logger().info(f"Received image message with latency {duration_latency}", throttle_duration_sec=1.0)
        self.on_messages_received_callback(time_message, message_image=message_image)

    def on_messages_received_callback(self, time_message, message_points=None, message_image=None, message_info=None):
        time_start = time.time()

        message_info = self.cache_info.getElemBeforeTime(time_message) if message_info is None else message_info
        message_points = self.cache_points.getElemBeforeTime(time_message) if message_points is None else message_points
        message_image = self.cache_image.getElemBeforeTime(time_message) if self.use_color_sampling and message_image is None else message_image

        if message_info is None or message_points is None or (self.use_color_sampling and message_image is None):
            self.get_logger().warning(f"Cache empty")
            return

        time_ros_info = Time.from_msg(message_info.header.stamp)
        time_ros_points = Time.from_msg(message_points.header.stamp)
        time_ros_image = Time.from_msg(message_image.header.stamp) if self.use_color_sampling else None

        self.lock.acquire()
        if any(time_ros_points == time_cached for time_cached in self.cache_times_points_message):
            self.get_logger().debug(f"Messages skipped because the pointcloud has been used already")
            self.lock.release()
            return
        self.cache_times_points_message.append(time_ros_points)
        self.lock.release()

        # TODO: Find out why latency of points message get high if some tasks are launched and re-activate this
        # duration_difference = abs((time_ros_info - time_ros_points).nanoseconds) / 1e9
        # if duration_difference > self.slop_synchronizer:
        #     self.get_logger().info(f"Camera info and  point-cloud stamps difference too big: {duration_difference}")
        #     return
        if self.use_color_sampling:
            duration_difference = abs((time_ros_points - time_ros_image).nanoseconds) / 1e9
            if duration_difference > self.slop_synchronizer:
                self.get_logger().warning(f"Image and point-cloud stamps difference too big: {duration_difference}")
                return

        if message_info.distortion_model == "rational_polynomial":
            model_camera = ModelRationalPolynomial.from_camera_info_message(message_info)
        elif message_info.distortion_model == "double_sphere":
            model_camera = ModelDoubleSphere.from_camera_info_message(message_info)
            if self.use_rectification:
                model_camera_original = model_camera
                model_camera = ModelRationalPolynomial(
                    fx=model_camera.fx,
                    fy=model_camera.fy,
                    cx=(model_camera.shape_image[2] / 2.0),
                    cy=(model_camera.shape_image[1] / 2.0),
                    model_distortion="rational_polynomial",
                    params_distortion=dict(k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0, k4=0.0, k5=0.0, k6=0.0),
                    shape_image=model_camera.shape_image,
                )

        is_success, info, message_points = self.tf_oracle.transform_to_frame(message_points, self.name_frame_camera)
        if not is_success:
            self.get_logger().warning(f"{info}")
            return
        # Methods like torch.frombuffer or np.frombuffer do not work if incoming data has points with padded bytes
        # E.g. the ouster/points topic has an itemsize of 16 while it publishes only xyz in float32
        pointcloud = point_cloud2.read_points(message_points, skip_nans=True)
        if self.use_color_sampling:
            image = self.bridge_cv.imgmsg_to_cv2(message_image, desired_encoding="passthrough")

        points = self.points2tensor(pointcloud)
        coords_uv_points, mask_valid = model_camera.project_points_onto_image(points, use_invalid_coords=True, use_mask_fov=True, use_half_precision=True)

        if self.use_color_sampling or self.use_rectification:
            images = self.image2tensor(image)
            if self.use_rectification:
                images = self.rectify_images(images, model_camera, model_camera_original)

        if self.use_color_sampling:
            pointcloud_colored, offset = self.compute_pointcloud_colored(coords_uv_points, images, pointcloud, mask_valid)
            self.publish_points(message_points, pointcloud_colored, offset)

        if self.use_depth_sampling:
            # Number of channels unknown
            self.sampler_depth.shape_image = model_camera.shape_image
            image_depth = self.compute_depth_image(coords_uv_points, points, mask_valid)

            self.publish_image_depth(image_depth, name_frame=message_info.header.frame_id, stamp=message_points.header.stamp)

        if self.use_rectification:
            image_rectified = images[0]
            image_rectified = image_rectified.permute(1, 2, 0).contiguous()
            image_rectified = np.ascontiguousarray(image_rectified.detach().cpu().numpy().astype(np.uint8))
            self.publish_image_rectified(image_rectified, name_frame=message_image.header.frame_id, stamp=message_image.header.stamp)

            info_rectified = dict(
                height=model_camera.shape_image[1],
                width=model_camera.shape_image[2],
                distortion_model=model_camera.model_distortion,
                d=list(model_camera.params_distortion.values()),
                k=[model_camera.fx, 0.0, model_camera.cx] + [0.0, model_camera.fy, model_camera.cy] + [0.0, 0.0, 1.0],
                r=[1.0, 0.0, 0.0] + [0.0, 1.0, 0.0] + [0.0, 0.0, 1.0],
                p=[model_camera.fx, 0.0, model_camera.cx, 0.0] + [0.0, model_camera.fy, model_camera.cy, 0.0] + [0.0, 0.0, 1.0, 0.0],
            )
            self.publish_info_rectified(info_rectified, name_frame=message_info.header.frame_id, stamp=message_info.header.stamp)

        time_end = time.time()
        duration_inference = time_end - time_start
        self.get_logger().info(f"Latency: {duration_inference:.3f}", throttle_duration_sec=1.0)

    def on_service_call_project_dome(self, request, response):
        response = ProjectDome.Response(success=True, message="")

        try:
            message_points = request.points if request.points.height != 0 or request.points.width != 0 else self.cache_points.getElemBeforeTime(self.get_clock().now())
            message_info = self.cache_info.getElemBeforeTime(Time.from_msg(message_points.header.stamp))

            is_success, message, message_points = self.tf_oracle.transform_to_frame(message_points, self.name_frame_camera)
            if not is_success:
                raise RuntimeError("Failed to transform points")

            pointcloud = point_cloud2.read_points(message_points, skip_nans=True)
            points = self.points2tensor(pointcloud)

            if message_info is not None:
                if message_info.distortion_model == "rational_polynomial":
                    model_camera = ModelRationalPolynomial.from_camera_info_message(message_info)
                elif message_info.distortion_model == "double_sphere":
                    model_camera = ModelDoubleSphere.from_camera_info_message(message_info)
            else:
                model_camera = ModelDoubleSphere(
                    self.info_double_sphere_default["fx"],
                    self.info_double_sphere_default["fy"],
                    self.info_double_sphere_default["cx"],
                    self.info_double_sphere_default["cy"],
                    model_distortion="double_sphere",
                    params_distortion=dict(xi=self.info_double_sphere_default["xi"], alpha=self.info_double_sphere_default["alpha"]),
                    shape_image=(-1, self.info_double_sphere_default["height"], self.info_double_sphere_default["width"]),
                )
            coords_uv_points, mask_valid = model_camera.project_points_onto_image(points, use_invalid_coords=True, use_mask_fov=True, use_half_precision=True)

            self.sampler_depth.shape_image = model_camera.shape_image
            image_depth = self.compute_depth_image(coords_uv_points, points, mask_valid)

            header = Header(stamp=message_points.header.stamp, frame_id=message_info.header.frame_id if message_info is not None else self.name_frame_camera)
            response.image = self.bridge_cv.cv2_to_imgmsg(image_depth, header=header, encoding="mono16")
        except Exception as e:
            response.success = False
            response.message = f"{e}"
            self.get_logger().error(f"Service: {response.message[:-1]}")

        return response

    def on_service_call_colorize_points(self, request, response):
        response = ColorizePoints.Response(success=True, message="")

        try:
            message_points = request.points if request.points.height != 0 or request.points.width != 0 else self.cache_points.getElemBeforeTime(self.get_clock().now())
            message_image = request.image if request.image.height != 0 or request.image.width != 0 else self.cache_image.getElemBeforeTime(self.get_clock().now())
            message_info = self.cache_info.getElemBeforeTime(Time.from_msg(message_points.header.stamp))

            is_success, message, message_points = self.tf_oracle.transform_to_frame(message_points, self.name_frame_camera)
            if not is_success:
                raise RuntimeError(f"{message}")

            pointcloud = point_cloud2.read_points(message_points, skip_nans=True)
            points = self.points2tensor(pointcloud)

            image = self.bridge_cv.imgmsg_to_cv2(message_image, desired_encoding="passthrough")
            images = self.image2tensor(image)

            if message_info is not None:
                if message_info.distortion_model == "rational_polynomial":
                    model_camera = ModelRationalPolynomial.from_camera_info_message(message_info)
                elif message_info.distortion_model == "double_sphere":
                    model_camera = ModelDoubleSphere.from_camera_info_message(message_info)
            else:
                model_camera = ModelDoubleSphere(
                    self.info_double_sphere_default["fx"],
                    self.info_double_sphere_default["fy"],
                    self.info_double_sphere_default["cx"],
                    self.info_double_sphere_default["cy"],
                    model_distortion="double_sphere",
                    params_distortion=dict(xi=self.info_double_sphere_default["xi"], alpha=self.info_double_sphere_default["alpha"]),
                    shape_image=(-1, self.info_double_sphere_default["height"], self.info_double_sphere_default["width"]),
                )
            coords_uv_points, mask_valid = model_camera.project_points_onto_image(points, use_invalid_coords=True, use_mask_fov=True, use_half_precision=True)

            pointcloud_colored, offset = self.compute_pointcloud_colored(coords_uv_points, images, pointcloud, mask_valid)

            fields = message_points.fields + [PointField(name="rgb", offset=offset, datatype=PointField.UINT32, count=1)]
            response.points_colored = point_cloud2.create_cloud(message_points.header, fields, pointcloud_colored)
        except Exception as e:
            response.success = False
            response.message = f"{e}"
            self.get_logger().error(f"Service: {response.message[:-1]}")

        return response

    def _init_parameters(self):
        self.add_on_set_parameters_callback(self.handler_parameters.parameter_callback)

        self._init_parameter_name_frame_camera()
        self._init_parameter_name_frame_lidar()
        self._init_parameter_topic_image()
        self._init_parameter_topic_image_rectified()
        self._init_parameter_topic_info()
        self._init_parameter_topic_info_rectified()
        self._init_parameter_topic_points()
        self._init_parameter_topic_projected_depth()
        self._init_parameter_topic_projected_points()
        self._init_parameter_slop_synchronizer()
        self._init_parameter_color_invalid()
        self._init_parameter_factor_downsampling()
        self._init_parameter_k_knn()
        self._init_parameter_mode_interpolation()
        self._init_parameter_use_color_sampling()
        self._init_parameter_use_depth_sampling()
        self._init_parameter_use_rectification()
        self._init_parameter_use_service_only()
        self._init_parameter_use_knn_interpolation()
        self._init_parameter_size_cache()

        self.handler_parameters.all_declared()

    def _init_parameter_name_frame_camera(self):
        descriptor = ParameterDescriptor(
            name="name_frame_camera",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the camera frame in the tf transform tree",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.name_frame_camera, descriptor)

    def _init_parameter_name_frame_lidar(self):
        descriptor = ParameterDescriptor(
            name="name_frame_lidar",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the lidar frame in the tf transform tree",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.name_frame_lidar, descriptor)

    def _init_parameter_topic_image(self):
        descriptor = ParameterDescriptor(
            name="topic_image",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the rgb image topic (for subscriber)",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.topic_image, descriptor)

    def _init_parameter_topic_image_rectified(self):
        descriptor = ParameterDescriptor(
            name="topic_image_rectified",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the rectified rgb image topic (for publisher)",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.topic_image_rectified, descriptor)

    def _init_parameter_topic_info(self):
        descriptor = ParameterDescriptor(
            name="topic_info",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the camera info topic (for subscriber)",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.topic_info, descriptor)

    def _init_parameter_topic_info_rectified(self):
        descriptor = ParameterDescriptor(
            name="topic_info_rectified",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the rectified camera info topic (for publisher)",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.topic_info_rectified, descriptor)

    def _init_parameter_topic_points(self):
        descriptor = ParameterDescriptor(
            name="topic_points",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the pointcloud topic (for subscriber)",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.topic_points, descriptor)

    def _init_parameter_topic_projected_depth(self):
        descriptor = ParameterDescriptor(
            name="topic_projected_depth",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the projected depth topic (for publisher)",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.topic_projected_depth, descriptor)

    def _init_parameter_topic_projected_points(self):
        descriptor = ParameterDescriptor(
            name="topic_projected_points",
            type=ParameterType.PARAMETER_STRING,
            description="Name of the colored pointcloud topic (for publisher)",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.topic_projected_points, descriptor)

    def _init_parameter_slop_synchronizer(self):
        descriptor = ParameterDescriptor(
            name="slop_synchronizer",
            type=ParameterType.PARAMETER_DOUBLE,
            description="Maximum time disparity between associated image and pointcloud messages",
            read_only=False,
            floating_point_range=(FloatingPointRange(from_value=0.0, to_value=2.0, step=0.0),),
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.slop_synchronizer, descriptor)

    def _init_parameter_color_invalid(self):
        descriptor = ParameterDescriptor(
            name="color_invalid",
            type=ParameterType.PARAMETER_STRING,
            description="Rgb color given to invalid points",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.color_invalid, descriptor)

    def _init_parameter_factor_downsampling(self):
        descriptor = ParameterDescriptor(
            name="factor_downsampling",
            type=ParameterType.PARAMETER_INTEGER,
            description="Downsampling factor used with knn interpolation",
            read_only=False,
            integer_range=(IntegerRange(from_value=1, to_value=16, step=1),),
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.factor_downsampling, descriptor)

    def _init_parameter_k_knn(self):
        descriptor = ParameterDescriptor(
            name="k_knn",
            type=ParameterType.PARAMETER_INTEGER,
            description="Number of neighbors used with knn interpolation",
            read_only=False,
            integer_range=(IntegerRange(from_value=1, to_value=10, step=1),),
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.k_knn, descriptor)

    def _init_parameter_mode_interpolation(self):
        descriptor = ParameterDescriptor(
            name="mode_interpolation",
            type=ParameterType.PARAMETER_STRING,
            description="Interpolation mode for upsampling used with knn interpolation",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.mode_interpolation, descriptor)

    def _init_parameter_use_color_sampling(self):
        descriptor = ParameterDescriptor(
            name="use_color_sampling",
            type=ParameterType.PARAMETER_BOOL,
            description="Usage of color sampling",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.use_color_sampling, descriptor)

    def _init_parameter_use_depth_sampling(self):
        descriptor = ParameterDescriptor(
            name="use_depth_sampling",
            type=ParameterType.PARAMETER_BOOL,
            description="Usage of depth sampling",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.use_depth_sampling, descriptor)

    def _init_parameter_use_rectification(self):
        descriptor = ParameterDescriptor(
            name="use_rectification",
            type=ParameterType.PARAMETER_BOOL,
            description="Usage of image rectification",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.use_rectification, descriptor)

    def _init_parameter_use_service_only(self):
        descriptor = ParameterDescriptor(
            name="use_service_only",
            type=ParameterType.PARAMETER_BOOL,
            description="Usage of service-only mode",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.use_service_only, descriptor)

    def _init_parameter_use_knn_interpolation(self):
        descriptor = ParameterDescriptor(
            name="use_knn_interpolation",
            type=ParameterType.PARAMETER_BOOL,
            description="Usage of knn interpolation",
            read_only=False,
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.use_knn_interpolation, descriptor)

    def _init_parameter_size_cache(self):
        descriptor = ParameterDescriptor(
            name="size_cache",
            type=ParameterType.PARAMETER_INTEGER,
            description="Cache size for synchronization",
            read_only=False,
            integer_range=(IntegerRange(from_value=1, to_value=50, step=1),),
        )
        self.parameter_descriptors.append(descriptor)
        self.declare_parameter(descriptor.name, self.size_cache, descriptor)

    # Rename call from parameter handler
    def parameter_changed(self, parameter):
        is_success, info = self.update_parameter(name=parameter.name, value=parameter.value)
        return is_success, info

    def update_parameter(self, name, value):
        try:
            func_update = getattr(self, f"update_{name}")
            is_success, info = func_update(value)
        except Exception as exception:
            is_success = False
            info = f"{exception}"
            self.get_logger().error(f"{exception}")

        return is_success, info

    def update_name_frame_camera(self, name_frame_camera):
        self.name_frame_camera = name_frame_camera

        is_success = True
        info = ""
        return is_success, info

    def update_name_frame_lidar(self, name_frame_lidar):
        self.name_frame_lidar = name_frame_lidar

        is_success = True
        info = ""
        return is_success, info

    def update_topic_image(self, topic_image):
        self._del_subscribers()
        self.topic_image = topic_image
        self._init_subscribers()

        is_success = True
        info = ""
        return is_success, info

    def update_topic_image_rectified(self, topic_image_rectified):
        self._del_publishers()
        self.topic_image_rectified = topic_image_rectified
        self._init_publishers()

        is_success = True
        info = ""
        return is_success, info

    def update_topic_info(self, topic_info):
        self._del_subscribers()
        self.topic_info = topic_info
        self._init_subscribers()

        is_success = True
        info = ""
        return is_success, info

    def update_topic_info_rectified(self, topic_info_rectified):
        self._del_publishers()
        self.topic_info_rectified = topic_info_rectified
        self._init_publishers()

        is_success = True
        info = ""
        return is_success, info

    def update_topic_points(self, topic_points):
        self._del_subscribers()
        self.topic_points = topic_points
        self._init_subscribers()

        is_success = True
        info = ""
        return is_success, info

    def update_topic_projected_depth(self, topic_projected_depth):
        self._del_publishers()
        self.topic_projected_depth = topic_projected_depth
        self._init_publishers()

        is_success = True
        info = ""
        return is_success, info

    def update_topic_projected_points(self, topic_projected_points):
        self._del_publishers()
        self.topic_projected_points = topic_projected_points
        self._init_publishers()

        is_success = True
        info = ""
        return is_success, info

    def update_slop_synchronizer(self, slop_synchronizer):
        self.slop_synchronizer = slop_synchronizer

        is_success = True
        info = ""
        return is_success, info

    def update_color_invalid(self, color_invalid):
        # TODO: This is a little dangerous. Find a better way.
        self.color_invalid = eval(color_invalid)
        self.sampler_color = SamplerColor(color_invalid=self.color_invalid)

        is_success = True
        info = ""
        return is_success, info

    def update_factor_downsampling(self, factor_downsampling):
        self.factor_downsampling = factor_downsampling
        self.sampler_depth = SamplerDepth(factor_downsampling=self.factor_downsampling, k_knn=self.k_knn, mode_interpolation=self.mode_interpolation)

        is_success = True
        info = ""
        return is_success, info

    def update_k_knn(self, k_knn):
        self.k_knn = k_knn
        self.sampler_depth = SamplerDepth(factor_downsampling=self.factor_downsampling, k_knn=self.k_knn, mode_interpolation=self.mode_interpolation)

        is_success = True
        info = ""
        return is_success, info

    def update_mode_interpolation(self, mode_interpolation):
        self.mode_interpolation = mode_interpolation
        self.sampler_depth = SamplerDepth(factor_downsampling=self.factor_downsampling, k_knn=self.k_knn, mode_interpolation=self.mode_interpolation)

        is_success = True
        info = ""
        return is_success, info

    def update_use_color_sampling(self, use_color_sampling):
        self._del_subscribers()
        self.use_color_sampling = use_color_sampling
        self._init_subscribers()

        is_success = True
        info = ""
        return is_success, info

    def update_use_depth_sampling(self, use_depth_sampling):
        self._del_subscribers()
        self.use_depth_sampling = use_depth_sampling
        self._init_subscribers()

        is_success = True
        info = ""
        return is_success, info

    def update_use_rectification(self, use_rectification):
        self._del_subscribers()
        self.use_rectification = use_rectification
        self._init_subscribers()

        is_success = True
        info = ""
        return is_success, info

    def update_use_service_only(self, use_service_only):
        self._del_subscribers()
        self.use_service_only = use_service_only
        self._init_subscribers()

        is_success = True
        info = ""
        return is_success, info

    def update_use_knn_interpolation(self, use_knn_interpolation):
        self.use_knn_interpolation = use_knn_interpolation

        is_success = True
        info = ""
        return is_success, info

    def update_size_cache(self, size_cache):
        self._del_subscribers()
        self.size_cache = size_cache
        self._init_subscribers()

        is_success = True
        info = ""
        return is_success, info
