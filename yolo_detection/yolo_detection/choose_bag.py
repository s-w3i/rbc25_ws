#!/usr/bin/env python3
"""
bag_detection_node.py

ROS2 node for bag detection using a YOLO model and RealSense camera input.
Acceptable classes: backpack, handbag, suitcase.
Displays detections in an OpenCV window with proper QoS for sensor data.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class BagDetectionNode(Node):
    def __init__(self):
        super().__init__('bag_detection_node')
        # Load the YOLO model
        self.model = YOLO('yolo11x.pt')
        # Define acceptable classes
        self.acceptable = {'backpack', 'handbag', 'suitcase'}
        # Initialize CvBridge
        self.bridge = CvBridge()
        # Subscribe to the RealSense color image topic with sensor data QoS (BEST_EFFORT reliability)
        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        self.subscription = self.create_subscription(
            Image,
            '/camera0/color/image_raw',
            self.image_callback,
            sensor_qos)
        # Create an OpenCV window for display
        cv2.namedWindow('bag_detection', cv2.WINDOW_NORMAL)
        self.get_logger().info('BagDetectionNode started and listening for images...')

    def image_callback(self, msg: Image):
        # Convert ROS Image to OpenCV image
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}')
            return

        # Perform detection
        results = self.model(frame)[0]

        # Iterate detections and draw boxes for acceptable classes
        for box in results.boxes:
            cls_id = int(box.cls.cpu().numpy())
            cls_name = results.names[cls_id]
            if cls_name in self.acceptable:
                # Extract coordinates and handle possible extra dimensions
                coords = box.xyxy.cpu().numpy().astype(int)
                # Flatten if nested
                coords = coords.flatten() if coords.ndim > 1 else coords
                x1, y1, x2, y2 = coords.tolist()
                conf = float(box.conf.cpu().numpy())
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put label text
                label = f'{cls_name} {conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('bag_detection', frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = BagDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
