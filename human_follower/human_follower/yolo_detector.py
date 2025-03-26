import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from robot_interfaces.msg import Detections, BoundingBox  # Import custom messages

class YOLODetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        self.bridge = CvBridge()
        
        # Subscribers
        self.color_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        
        # Publisher
        self.detection_pub = self.create_publisher(Detections, '/detections', 10)
        
        # Load YOLO model
        self.yolo = YOLO('yolov8n.pt')  # YOLOv8 model
        
        # Video writer to save output
        self.video_writer = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
        
        # Store the latest depth image
        self.depth_image = None

    def depth_callback(self, msg):
        # Convert ROS depth image to OpenCV image
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def color_callback(self, msg):
        if self.depth_image is None:
            self.get_logger().warn("No depth image received yet.")
            return
        
        # Convert ROS color image to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # Detect humans using YOLO
        results = self.yolo(cv_image)
        detections = Detections()
        
        for result in results:
            for box in result.boxes:
                if box.cls == 0:  # Class 0 is 'person'
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf.item()
                    
                    # Calculate the center of the bounding box
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Get the depth value at the center (in meters)
                    depth = self.depth_image[center_y, center_x] / 1000.0  # Convert mm to meters
                    
                    # Create a BoundingBox message
                    detection = BoundingBox()
                    detection.x1 = x1
                    detection.y1 = y1
                    detection.x2 = x2
                    detection.y2 = y2
                    detection.confidence = conf
                    detection.distance = depth  # Add distance to the message
                    detections.detections.append(detection)
        
        # Publish detections
        self.detection_pub.publish(detections)
        
        # Visualize detections
        for detection in detections.detections:
            cv2.rectangle(cv_image, (detection.x1, detection.y1), (detection.x2, detection.y2), (0, 255, 0), 2)
            cv2.putText(cv_image, f"{detection.distance:.2f}m", (detection.x1, detection.y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow("YOLO Detections", cv_image)
        cv2.waitKey(1)
        
        # Save the frame to the video file
        self.video_writer.write(cv_image)

    def __del__(self):
        # Release the video writer when the node is destroyed
        self.video_writer.release()

def main(args=None):
    rclpy.init(args=args)
    yolo_detector = YOLODetector()
    rclpy.spin(yolo_detector)
    yolo_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()