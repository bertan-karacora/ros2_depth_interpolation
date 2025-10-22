from ros2_depth_interpolation.ros import NodeInterpolationDepth

import ros2_utils.node as utils_node


def main():
    utils_node.start_and_spin_node(NodeInterpolationDepth)


if __name__ == "__main__":
    main()
