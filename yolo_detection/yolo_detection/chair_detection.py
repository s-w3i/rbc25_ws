#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_srvs.srv import Trigger
from ultralytics import YOLO
from robot_interfaces.msg import Detections, BoundingBox  # Custom messages

class YOLOEmptyChairService(Node):
    def __init__(self):
        super().__init__('yolo_empty_chair_service')
        self.bridge = CvBridge()

        # Subscribers to store the latest images and camera info.
        self.color_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)

        # Service to trigger detection.
        self.srv = self.create_service(Trigger, 'detect_empty_chair', self.service_callback)

        # Publisher for empty chair detections.
        self.detection_pub = self.create_publisher(Detections, '/empty_chair_detections', 10)

        # Load YOLO segmentation model (e.g., YOLOv8-seg).
        self.yolo = YOLO('yolov8n-seg.pt')
        self.get_logger().info(f"YOLO segmentation model loaded. Model names: {self.yolo.names}")

        # Latest data storage.
        self.latest_color = None
        self.latest_depth = None
        self.latest_camera_info = None

    def color_callback(self, msg):
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error("Error converting color image: " + str(e))

    def depth_callback(self, msg):
        try:
            # Using passthrough encoding so we keep the native format.
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error("Error converting depth image: " + str(e))

    def camera_info_callback(self, msg):
        self.latest_camera_info = msg

    def service_callback(self, request, response):
        # Ensure all data is available.
        if self.latest_color is None or self.latest_depth is None or self.latest_camera_info is None:
            response.success = False
            response.message = "Waiting for complete camera data (color, depth, camera info)."
            return response

        # Copy the current data.
        cv_image = self.latest_color.copy()
        depth_image = self.latest_depth
        camera_info = self.latest_camera_info

        # Run YOLO segmentation on the captured color image.
        results = self.yolo(cv_image)
        result = results[0]  # Assuming a single result
        
        # Separate detections for chairs and persons.
        chairs = []
        persons = []
        for i, box in enumerate(result.boxes):
            cls = int(box.cls)
            label = self.yolo.names[cls].lower()
            bbox = list(map(int, box.xyxy[0]))  # [x1, y1, x2, y2]
            conf = box.conf.item()
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            mask = None
            if result.masks is not None and result.masks.data is not None:
                # Get the segmentation mask and threshold it.
                mask_np = result.masks.data[i].cpu().numpy() if hasattr(result.masks.data[i], 'cpu') else result.masks.data[i]
                mask = (mask_np > 0.5).astype(np.uint8)
            
            detection_info = {'bbox': bbox, 'conf': conf, 'center': center, 'mask': mask}
            if label == 'chair':
                chairs.append(detection_info)
            elif label == 'person':
                persons.append(detection_info)

        # Determine if each chair is occupied by checking mask overlap.
        occupancy_threshold = 0.3  # Adjust as needed.
        empty_chairs = []
        for chair in chairs:
            is_occupied = False
            if chair['mask'] is not None:
                chair_mask = chair['mask']
                for person in persons:
                    if person['mask'] is None:
                        continue
                    person_mask = person['mask']
                    intersection = np.logical_and(chair_mask, person_mask)
                    overlap_ratio = np.sum(intersection) / np.sum(chair_mask) if np.sum(chair_mask) > 0 else 0
                    if overlap_ratio > occupancy_threshold:
                        is_occupied = True
                        break
            if not is_occupied:
                empty_chairs.append(chair)

        # Compute 3D coordinates for each empty chair and select the nearest one.
        nearest_empty_chair = None
        nearest_distance = float('inf')
        nearest_coords = None
        for chair in empty_chairs:
            center = chair['center']
            # Get the depth value at the center (assume depth in millimeters; convert to meters).
            depth_val = depth_image[center[1], center[0]] / 1000.0
            if depth_val <= 0:
                continue
            # Retrieve camera intrinsics.
            fx = camera_info.k[0]
            fy = camera_info.k[4]
            cx_intr = camera_info.k[2]
            cy_intr = camera_info.k[5]
            # Compute 3D coordinates using the pinhole camera model.
            x_3d = (center[0] - cx_intr) * depth_val / fx
            y_3d = (center[1] - cy_intr) * depth_val / fy
            z_3d = depth_val
            distance = np.sqrt(x_3d**2 + y_3d**2 + z_3d**2)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_empty_chair = chair
                nearest_coords = (x_3d, y_3d, z_3d)

        # Prepare a detection message to publish.
        detections_msg = Detections()
        if nearest_empty_chair is not None:
            bbox = nearest_empty_chair['bbox']
            detection = BoundingBox()
            detection.x1, detection.y1, detection.x2, detection.y2 = bbox
            detection.confidence = nearest_empty_chair['conf']
            detection.distance = nearest_distance  # Euclidean distance (in meters)
            detections_msg.detections.append(detection)
            
            # Draw the bounding box on the image.
            cv2.rectangle(cv_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            if nearest_coords is not None:
                text = f"x:{nearest_coords[0]:.2f}m, y:{nearest_coords[1]:.2f}m, z:{nearest_coords[2]:.2f}m"
                cv2.putText(cv_image, text, (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            response.success = True
            response.message = f"Nearest empty chair: {text}"
        else:
            response.success = False
            response.message = "No empty chair detected."

        # Publish the detection message.
        self.detection_pub.publish(detections_msg)
        
        # Save the detection result image to /home/usern.
        save_path = "/home/usern/empty_chair_detection.jpg"
        cv2.imwrite(save_path, cv_image)
        self.get_logger().info(f"Detection image saved to {save_path}")

        # Show the detection result in a window.
        cv2.imshow("Empty Chair Detection", cv_image)
        cv2.waitKey(1)
        return response

def main(args=None):
    rclpy.init(args=args)
    node = YOLOEmptyChairService()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
