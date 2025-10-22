from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node


def generate_launch_description():
    args_launch = [
        DeclareLaunchArgument("name_frame_camera", description="Name of the camera frame in the tf transform tree", default_value="camera_ids_link"),
        DeclareLaunchArgument("name_frame_lidar", description="Name of the lidar frame in the tf transform tree", default_value="os_sensor_link"),
        DeclareLaunchArgument("topic_image", description="Name of the rgb image topic (for subscriber)", default_value="/camera_ids/image_color"),
        DeclareLaunchArgument("topic_image_rectified", description="Name of the rectified rgb image topic (for publisher)", default_value="/camera_ids/rectified/image_color"),
        DeclareLaunchArgument("topic_info", description="Name of the camera info topic (for subscriber)", default_value="/camera_ids/camera_info"),
        DeclareLaunchArgument("topic_info_rectified", description="Name of the rectified camera info topic (for publisher)", default_value="/camera_ids/rectified/camera_info"),
        DeclareLaunchArgument("topic_points", description="Name of the pointcloud topic (for subscriber)", default_value="/ouster/points"),
        DeclareLaunchArgument("topic_projected_depth", description="Name of the projected depth topic (for publisher)", default_value="/camera_ids/projected/depth/image"),
        DeclareLaunchArgument("topic_projected_points", description="Name of the colored pointcloud topic (for publisher)", default_value="/ouster/projected/points"),
        DeclareLaunchArgument("slop_synchronizer", description="Maximum time disparity between associated image and pointcloud messages", default_value="0.1"),
        DeclareLaunchArgument("color_invalid", description="Rgb color given to invalid points", default_value="(255, 87, 51)"),
        DeclareLaunchArgument("factor_downsampling", description="Downsampling factor used with knn interpolation", default_value="8"),
        DeclareLaunchArgument("k_knn", description="Number of neighbors used with knn interpolation", default_value="1"),
        DeclareLaunchArgument(
            "mode_interpolation",
            description="Interpolation mode for upsampling used with knn interpolation",
            choices=["nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"],
            default_value="nearest",
        ),
        DeclareLaunchArgument("use_depth_sampling", description="Usage of depth sampling", choices=["True", "False"], default_value="True"),
        DeclareLaunchArgument("use_color_sampling", description="Usage of color sampling", choices=["True", "False"], default_value="True"),
        DeclareLaunchArgument("use_rectification", description="Usage of image rectification", choices=["True", "False"], default_value="False"),
        DeclareLaunchArgument("use_service_only", description="Usage of service-only mode", choices=["True", "False"], default_value="False"),
        DeclareLaunchArgument("use_knn_interpolation", description="Usage of knn interpolation", choices=["True", "False"], default_value="True"),
        DeclareLaunchArgument("size_cache", description="Cache size for synchronization", default_value="10"),
    ]
    launch_description = LaunchDescription(args_launch)

    action_depth_interpolatio = Node(
        package="ros2_depth_interpolation",
        namespace="",
        executable="spin",
        name="ros2_depth_interpolation",
        output="screen",
        parameters=[
            {
                "name_frame_camera": LaunchConfiguration("name_frame_camera"),
                "name_frame_lidar": LaunchConfiguration("name_frame_lidar"),
                "topic_image": LaunchConfiguration("topic_image"),
                "topic_image_rectified": LaunchConfiguration("topic_image_rectified"),
                "topic_info": LaunchConfiguration("topic_info"),
                "topic_info_rectified": LaunchConfiguration("topic_info_rectified"),
                "topic_points": LaunchConfiguration("topic_points"),
                "topic_projected_depth": LaunchConfiguration("topic_projected_depth"),
                "topic_projected_points": LaunchConfiguration("topic_projected_points"),
                "slop_synchronizer": LaunchConfiguration("slop_synchronizer"),
                "color_invalid": LaunchConfiguration("color_invalid"),
                "factor_downsampling": LaunchConfiguration("factor_downsampling"),
                "k_knn": LaunchConfiguration("k_knn"),
                "mode_interpolation": LaunchConfiguration("mode_interpolation"),
                "use_depth_sampling": LaunchConfiguration("use_depth_sampling"),
                "use_color_sampling": LaunchConfiguration("use_color_sampling"),
                "use_rectification": LaunchConfiguration("use_rectification"),
                "use_service_only": LaunchConfiguration("use_service_only"),
                "use_knn_interpolation": LaunchConfiguration("use_knn_interpolation"),
                "size_cache": LaunchConfiguration("size_cache"),
            }
        ],
        respawn=True,
        respawn_delay=5.0,
    )
    launch_description.add_action(action_depth_interpolatio)

    return launch_description
