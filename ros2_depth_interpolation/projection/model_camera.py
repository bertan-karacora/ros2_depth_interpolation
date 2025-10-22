import torch


class ModelCamera:
    """Base class for camera models."""

    def __init__(self, fx, fy, cx, cy, model_distortion, params_distortion, shape_image):
        self.cx = cx
        self.cy = cy
        self.device = None
        self.fx = fx
        self.fy = fy
        self.model_distortion = model_distortion
        self.params_distortion = params_distortion
        self.shape_image = shape_image

        self._init()

    def _init(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def from_camera_info_message(cls, message):
        """Create an instance from a camera info message."""
        try:
            binning_x = message.binning_x if message.binning_x != 0 else 1
            binning_y = message.binning_y if message.binning_y != 0 else 1
        except AttributeError:
            binning_x = 1
            binning_y = 1

        try:
            offset_x = message.roi.offset_x
            offset_y = message.roi.offset_y
            # Do not know channel dimension from camera info message but keep it for pytorch-like style
            shape_image = (-1, message.roi.height, message.roi.width)
        except AttributeError:
            offset_x = 0
            offset_y = 0
            shape_image = (-1, message.height, message.width)

        fx = message.k[0] / binning_x
        fy = message.k[4] / binning_y
        cx = (message.k[2] - offset_x) / binning_x
        cy = (message.k[5] - offset_y) / binning_y

        model_distortion = message.distortion_model
        params_distortion = cls.create_dict_params_distortion(message.d)

        instance = cls(fx, fy, cx, cy, model_distortion, params_distortion, shape_image)
        return instance

    @classmethod
    def create_dict_params_distortion(cls, list_params_distortion):
        try:
            params_distortion = dict(zip(cls.keys_params_distortion, list_params_distortion))
        except AttributeError:
            params_distortion = dict(enumerate(list_params_distortion))

        return params_distortion

    def project_points_onto_image(self, coords_xyz):
        """Project 3D points onto 2D image.
        Shape of coords_xyz: (B, 3, N)
        Coordinate frame of points: [right, down, front]
        Coordinate frame of image: [right, down]"""
        raise NotImplementedError()

    def project_image_onto_points(self, coords_uv):
        """Project 2D image onto 3D unit sphere.
        Shape of coords_uv: (B, 2, N)
        Coordinate frame of points: [right, down, front]
        Coordinate frame of image: [right, down]"""
        raise NotImplementedError()
