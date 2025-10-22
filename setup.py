import glob
import setuptools

NAME_PACKAGE = "ros2_depth_interpolation"
VERSION_PACKAGE = "0.0.1"
DESCRIPTION_PACKAGE = "ROS 2 package for dense depth interpolation and point-cloud coloring"
MAINTAINER_PACKAGE = "Bertan Karacora"
EMAIL_MAINTAINER_PACKAGE = "bertan.karacora@gmail.com"

setuptools.setup(
    name=NAME_PACKAGE,
    version=VERSION_PACKAGE,
    description=DESCRIPTION_PACKAGE,
    license="MIT",
    maintainer=MAINTAINER_PACKAGE,
    maintainer_email=EMAIL_MAINTAINER_PACKAGE,
    packages=setuptools.find_namespace_packages(exclude=["container", "docs", "launch", "libs", "notebooks", "resource", "resources", "scripts"]),
    install_requires=["setuptools"],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{NAME_PACKAGE}"]),
        (f"share/{NAME_PACKAGE}", ["package.xml"]),
        (f"share/{NAME_PACKAGE}/launch", glob.glob("launch/*.py")),
    ],
    package_data={NAME_PACKAGE: ["configs/*"]},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            f"spin = {NAME_PACKAGE}.scripts.spin_interpolation:main",
        ],
    },
)
