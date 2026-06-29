#!/usr/bin/env python3
"""
Standalone demo for the drone object detection and tracking pipeline.

Exercises the same YOLO-based inference and tracking logic as detector_node.py,
without requiring a live ROS2 environment. Detection output is printed in the
structure that would be published to /drone/detections.

Usage:
    python demo.py                  # webcam
    python demo.py --source 0       # explicit webcam index
    python demo.py --source video.mp4
"""

import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO


COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    14: "bird", 15: "cat", 16: "dog", 24: "backpack", 26: "handbag",
    39: "bottle", 41: "cup", 56: "chair", 57: "couch", 58: "potted plant",
    60: "dining table", 63: "laptop", 67: "cell phone",
}


def format_detection_msg(box, frame_id="drone_camera"):
    """Mirror the Detection2D message fields from detector_node.py."""
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    track_id = int(box.id[0]) if box.id is not None else None
    label = COCO_NAMES.get(cls_id, str(cls_id))

    return {
        "frame_id": frame_id,
        "bbox": {"center": {"x": round(cx, 1), "y": round(cy, 1)},
                 "size_x": round(w, 1), "size_y": round(h, 1)},
        "class_id": cls_id,
        "class_name": label,
        "score": round(conf, 3),
        "track_id": track_id,
    }


def draw_detections(frame, results):
    """Draw bounding boxes, class labels, track IDs, and confidence."""
    if results[0].boxes is None:
        return frame

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        track_id = int(box.id[0]) if box.id is not None else None
        label = COCO_NAMES.get(cls_id, str(cls_id))

        color = (0, 200, 0) if track_id is not None else (200, 100, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        tag = f"#{track_id} " if track_id is not None else ""
        text = f"{tag}{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return frame


def run(source=0):
    model = YOLO("yolov8n.pt")
    print(f"[INFO] YOLOv8n loaded. Opening source: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source '{source}'. "
              "Try a different --source value or connect a webcam.")
        return

    fps_timer = time.time()
    frame_count = 0
    fps = 0.0

    print("[INFO] Running detection. Press 'q' to quit.")
    print("[INFO] Simulating /drone/detections topic output:\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Stream ended.")
            break

        results = model.track(frame, persist=True, verbose=False)

        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                detections.append(format_detection_msg(box))

        # Print detection data (mirrors what detector_node publishes to ROS2)
        if detections:
            print(f"[/drone/detections] {len(detections)} object(s) detected:")
            for d in detections:
                tid = f"track_id={d['track_id']}" if d["track_id"] is not None else "untracked"
                print(f"  {d['class_name']} | conf={d['score']} | "
                      f"cx={d['bbox']['center']['x']} cy={d['bbox']['center']['y']} | {tid}")

        frame = draw_detections(frame, results)

        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_timer = time.time()

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Drone Vision Pipeline | YOLOv8 + Tracking", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("Drone Object Detection & Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Demo complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone vision pipeline demo")
    parser.add_argument("--source", default=0,
                        help="Video source: webcam index (0) or path to video file")
    args = parser.parse_args()

    source = args.source
    if source != 0:
        try:
            source = int(source)
        except ValueError:
            pass  # treat as file path

    run(source)
