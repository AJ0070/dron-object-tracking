from setuptools import setup

package_name = 'drone_vision'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jash Mann',
    maintainer_email='jash@example.com',
    description='Drone object detection and tracking with YOLO',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector_node = drone_vision.detector_node:main',
        ],
    },
)
