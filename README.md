# ROS 2 Depth Interpolation

ROS 2 package for depth interpolation and point cloud coloring.

Note: This is a public, stripped-down version of a private repository. It may depend on other repositories which might not have a public version. Some paths, configurations, dependencies, have been removed or altered, so the code may not run out of the box.

| ![docs/images/screenshot1.png](docs/images/screenshot1.png) | ![docs/images/screenshot2.png](docs/images/screenshot2.png) |
| :---------------------------------------------------------: | :---------------------------------------------------------: |

*Figure 1: Point cloud coloring in real-time. Orange points are outside of the camera's FOV.*

![docs/images/screenshot3.png](docs/images/screenshot3.png)
*Figure 2: The camera is placed a little higher than the 3D LiDAR, so there are artifacts if objects occlude the measurements of one of the sensors, see the borders of the table.*

| ![docs/images/screenshot4.png](docs/images/screenshot4.png) | ![docs/images/screenshot5.png](docs/images/screenshot5.png) |
| :---------------------------------------------------------: | :---------------------------------------------------------: |

*Figure 3: Dense depth estimation for the fisheye lense camera using knn interpolation on point cloud from 3D LiDAR.*

|                        ![docs/images/image1.png](docs/images/image1.jpg)                        |                     ![docs/images/image4.png](docs/images/image4.jpg)                     |
| :---------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------: |
| *Figure 4: Comparison of interpolated depth and measured depth using a stereo IR depth camera.* | *Figure 5: Rectification and pixel-aligned depth from interpolation. Using an large FOV.* |

| ![docs/images/image2.png](docs/images/image2.jpg) | ![docs/images/image3.png](docs/images/image3.jpg) |
| :-----------------------------------------------: | :-----------------------------------------------: |
|          *Figure 6: Using a medium FOV.*          |          *Figure 7: Using an small FOV.*          |

## Setup

```bash
git clone https://github.com/bertan-karacora/ros2_depth_interpolation.git
cd ros2_depth_interpolation
colcon build --packages-select ros2_depth_interpolation
```

## Installation

### Build container

```bash
container/build.sh
```

## Usage

```bash
ros2 launch ros2_depth_interpolation depth_interpolation_launch.py
```

## Links

- [The Double Sphere Camera Model](https://arxiv.org/pdf/1807.08957)

## TODO

- Disantagle ROS 2 interface and interpolation pipeline
