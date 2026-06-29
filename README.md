# Drone Object Detection and Tracking with ROS2

A drone vision pipeline for real-time object detection and tracking from aerial camera feeds using YOLO-based inference, integrated with ROS2 for robotics applications.

## Features

- YOLO-based object detection from drone camera feeds
- Real-time object tracking across frames
- ROS2 integration for downstream robotics workflows
- Published detections via ROS2 topics

## Requirements

- ROS2 (Humble or later)
- Python 3.8+
- OpenCV
- ultralytics (YOLOv8)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Build ROS2 package
colcon build --packages-select drone_vision
source install/setup.bash
```

## Usage

```bash
# Launch the detection node
ros2 run drone_vision detector_node

# View detections
ros2 topic echo /drone/detections
```

## Topics

- `/drone/detections` - Detected objects with bounding boxes and confidence scores
- `/drone/tracked_objects` - Tracked objects with IDs across frames
