#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.qos import QoSPresetProfiles
from ultralytics import YOLO
from robot_interfaces.srv import SegmentHumans
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from builtin_interfaces.msg import Time as TimeMsg
import tf2_ros
import tf2_geometry_msgs  # register PoseStamped transform support

class HumanSegmentationNode(Node):
    def __init__(self):
        super().__init__('human_segmentation_node')
        self.bridge = CvBridge()
        qos = QoSPresetProfiles.SENSOR_DATA.value

        # Subscribers
        self.color_sub = self.create_subscription(
            Image, '/camera0/color/image_raw', self.color_callback, qos)
        self.depth_sub = self.create_subscription(
            Image, '/camera0/depth/image_rect_raw', self.depth_callback, qos)
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera0/color/camera_info', self.info_callback, qos)

        # Service
        self.srv = self.create_service(
            SegmentHumans, 'segment_humans', self.service_callback)

        # Load YOLO segmentation model
        self.yolo = YOLO('yolov8n-seg.pt')
        self.get_logger().info(f"Model loaded with classes: {self.yolo.names}")

        # TF2 buffer with custom cache time
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=1.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.person_tfs: list[TransformStamped] = []
        self.create_timer(0.1, self._broadcast_person_tfs)
        self.tf_pub       = self.create_publisher(TransformStamped, "/people_transform", 10)

        # Storage for latest messages
        self.latest_color = None
        self.latest_depth = None
        self.latest_info = None

    def color_callback(self, msg: Image):
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert color image: {e}")

    def depth_callback(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            if depth.dtype == np.uint16:          # mm → m
                depth = depth.astype(np.float32) / 1000.0
            self.latest_depth = depth
        except Exception as e:
            self.get_logger().error(f"Depth conversion error: {e}")

    def info_callback(self, msg: CameraInfo):
        self.latest_info = msg

    def service_callback(self, request, response):
        self.person_tfs.clear()
        # Validate inputs
        if self.latest_color is None or self.latest_depth is None or self.latest_info is None:
            response.success = False
            response.message = "Waiting for sensor data (color, depth, camera_info)."
            return response

        # Run YOLO segmentation
        results = self.yolo(self.latest_color, conf=0.7)
        if not results or results[0].masks is None:
            response.success = False
            response.message = "No segmentation masks detected."
            return response

        # Extract masks, classes, confidences
        masks = results[0].masks.data
        if hasattr(masks, 'cpu'):
            masks = masks.cpu().numpy()
        classes = results[0].boxes.cls
        if hasattr(classes, 'cpu'):
            classes = classes.cpu().numpy()
        confs = results[0].boxes.conf
        if hasattr(confs, 'cpu'):
            confs = confs.cpu().numpy()

        segmented_images = []
        centroids = []
        K = self.latest_info.k
        fx, fy, cx_cam, cy_cam = K[0], K[4], K[2], K[5]
        now_msg = self.get_clock().now().to_msg()

        # Process each detection
        for i, mask in enumerate(masks):
            if confs[i] < 0.7:
                continue
            cls_idx = int(classes[i])
            class_name = (self.yolo.names[cls_idx]
                          if isinstance(self.yolo.names, (list, tuple))
                          else self.yolo.names.get(cls_idx))
            if class_name != "person":
                continue

            # Full-frame binary mask
            binary = (mask > 0.5).astype(np.uint8) * 255
            if cv2.countNonZero(binary) == 0:
                continue

            # Compute 2D centroid
            M = cv2.moments(binary)
            if M['m00'] == 0:
                continue
            cx2d = int(M['m10'] / M['m00'])
            cy2d = int(M['m01'] / M['m00'])

            # Depth at centroid
            if not (0 <= cy2d < self.latest_depth.shape[0] and 0 <= cx2d < self.latest_depth.shape[1]):
                continue
            depth_val = float(self.latest_depth[cy2d, cx2d])

            # Back-project to 3D in camera frame
            X = (cx2d - cx_cam) * depth_val / fx
            Y = (cy2d - cy_cam) * depth_val / fy
            Z = depth_val
            centroids.append(Point(x=X, y=Y, z=Z))

            # Broadcast TF for this person
            pose_raw = PoseStamped()
            pose_raw.header.stamp = now_msg
            pose_raw.header.frame_id = 'camera0_color_optical_frame'
            pose_raw.pose.position.x = X
            pose_raw.pose.position.y = Y
            pose_raw.pose.position.z = Z
            pose_raw.pose.orientation.w = 1.0

            # Transform into map frame (zero-stamp for latest)
            pose_tf = PoseStamped()
            pose_tf.header.stamp = TimeMsg()
            pose_tf.header.frame_id = pose_raw.header.frame_id
            pose_tf.pose = pose_raw.pose
            try:
                map_pose = self.tf_buffer.transform(pose_tf, 'map')
                map_pose.pose.position.z = 0.0  # Flatten Z
                tf_msg = TransformStamped()
                tf_msg.header.stamp = now_msg
                tf_msg.header.frame_id = 'map'
                tf_msg.child_frame_id = f"person_{i}"
                tf_msg.transform.translation.x = map_pose.pose.position.x
                tf_msg.transform.translation.y = map_pose.pose.position.y
                tf_msg.transform.translation.z = map_pose.pose.position.z
                tf_msg.transform.rotation = map_pose.pose.orientation
                self.tf_broadcaster.sendTransform(tf_msg)
                self.person_tfs.append(tf_msg)
                self.tf_pub.publish(tf_msg)
            except Exception as e:
                self.get_logger().warn(f"TF transform failed for person {i}: {e}")

            # Create segmented image mask
            mask_3ch = cv2.merge([binary, binary, binary])
            segmented_color = cv2.bitwise_and(self.latest_color, mask_3ch)
            try:
                img_msg = self.bridge.cv2_to_imgmsg(segmented_color, 'bgr8')
                segmented_images.append(img_msg)
            except Exception as e:
                self.get_logger().warn(f"Failed to convert segmented image: {e}")

        # Prepare response
        if not segmented_images:
            response.success = False
            response.message = "No person segmented."
        else:
            response.success = True
            response.message = f"{len(segmented_images)} person(s) segmented and TF published."
        response.segmented_images = segmented_images
        response.centroids = centroids
        return response

    def _broadcast_person_tfs(self):
            """Re-publish all stored person_tf messages at the current time."""
            now = self.get_clock().now().to_msg()
            for tf_msg in self.person_tfs:
                tf_msg.header.stamp = now
                self.tf_broadcaster.sendTransform(tf_msg)

def main(args=None):
    rclpy.init(args=args)
    node = HumanSegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
