from setuptools import find_packages, setup
import os
from glob import glob

package_name = "video"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Francois",
    maintainer_email="francois.proulx@usherbrooke.ca",
    description="Video nodes for recording and replay",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "video_replay = video.video_replay:main",
            "video_quat = video.video_quat:main",
        ],
    },
)
