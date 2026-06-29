#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO


class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector_node')

        self.model = YOLO('yolov8n.pt')
        self.bridge = CvBridge()
        self.tracker = {}
        self.next_id = 0

        self.subscription = self.create_subscription(
            Image,
            '/drone/camera/image_raw',
            self.image_callback,
            10)

        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/drone/detections',
            10)

        self.get_logger().info('Detector node initialized')

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        results = self.model.track(frame, persist=True, verbose=False)

        detection_array = Detection2DArray()
        detection_array.header = Header()
        detection_array.header.stamp = self.get_clock().now().to_msg()
        detection_array.header.frame_id = msg.header.frame_id

        if results[0].boxes is not None:
            for box in results[0].boxes:
                detection = Detection2D()

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detection.bbox.center.position.x = float((x1 + x2) / 2)
                detection.bbox.center.position.y = float((y1 + y2) / 2)
                detection.bbox.size_x = float(x2 - x1)
                detection.bbox.size_y = float(y2 - y1)

                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = str(int(box.cls[0]))
                hypothesis.hypothesis.score = float(box.conf[0])
                detection.results.append(hypothesis)

                if box.id is not None:
                    detection.id = str(int(box.id[0]))

                detection_array.detections.append(detection)

        self.detection_pub.publish(detection_array)


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
