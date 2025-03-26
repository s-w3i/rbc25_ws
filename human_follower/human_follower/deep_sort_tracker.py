#!/usr/bin/env python3
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import numpy as np

# Import tf2 related libraries and PoseStamped for goal publishing
from geometry_msgs.msg import TransformStamped, PoseStamped
import tf2_ros
from tf2_geometry_msgs import do_transform_pose

class RealsenseTrackerNode(Node):
    def __init__(self):
        super().__init__("realsense_tracker_node")
        self.bridge = CvBridge()

        # Initialize YOLOv8n model (lightweight)
        self.model = YOLO("yolov8n.pt")  # or "yolov8n.onnx"

        # Initialize Deep SORT tracker
        self.tracker = DeepSort(
            max_age=10000,              # Adjust for crowded or fast-moving scenes
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.4,
            nn_budget=5000,
            override_track_class=None,
            embedder="torchreid", 
            embedder_model_name="osnet_x1_0",   # Lightweight embedder
            half=True,               # Use FP16 if your GPU supports it
            bgr=True,
            embedder_gpu=True,
        )

        # Initialize tf2 broadcaster, buffer, and listener
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publisher for goal pose for nav2
        self.goal_publisher = self.create_publisher(PoseStamped, "/goal_pose", 10)

        # Subscribe to RealSense color image topic
        self.image_subscriber = self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",  # Adjust if your topic name differs
            self.image_callback,
            10
        )

        # Subscribe to RealSense depth image topic
        self.depth_subscriber = self.create_subscription(
            Image,
            "/camera/camera/depth/image_rect_raw",  # Adjust this topic name if needed
            self.depth_callback,
            10
        )

        self.latest_depth = None  # Store the latest depth image

        # Camera intrinsics (adjust these values to your camera's calibration)
        self.fx = 382.2181396484375  
        self.fy = 381.877197265625
        self.cx = 316.27850341796875  
        self.cy = 241.29092407226562

        self.get_logger().info("RealsenseTrackerNode initialized and subscribing to color and depth topics")

    def depth_callback(self, msg: Image):
        """Callback to receive depth image messages."""
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            # If depth is in mm, convert to meters:
            if depth_image.dtype == np.uint16:
                depth_image = depth_image.astype(np.float32) / 1000.0
            self.latest_depth = depth_image
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")

    def image_callback(self, msg: Image):
        """ROS callback for color images from the RealSense camera."""
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Perform YOLOv8 inference on 'person' class only (class index 0 in COCO)
        results = self.model.predict(frame, classes=[0], verbose=False, conf=0.5)

        # Extract bounding boxes (x, y, w, h) in XYWH format and confidence
        detections_xywh = results[0].boxes.xywh.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        # Prepare a list of detections: [[x1, y1, w, h], confidence, class]
        detections_list = []
        for det_xywh, conf in zip(detections_xywh, confidences):
            x_c, y_c, w, h = det_xywh
            x1 = x_c - w / 2
            y1 = y_c - h / 2
            detections_list.append([[x1, y1, w, h], conf, 0])

        # Update the Deep SORT tracker
        tracks = self.tracker.update_tracks(detections_list, frame=frame)

        # Process each confirmed track
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()  # left, top, right, bottom

            # Draw bounding box and track ID on the frame
            cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])),
                          (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (int(ltrb[0]), int(ltrb[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Compute the center pixel of the bounding box
            center_u = int((ltrb[0] + ltrb[2]) / 2)
            center_v = int((ltrb[1] + ltrb[3]) / 2)

            # Default transformation values
            x_cam, y_cam, z_cam = float('nan'), float('nan'), float('nan')
            if self.latest_depth is not None:
                if 0 <= center_v < self.latest_depth.shape[0] and 0 <= center_u < self.latest_depth.shape[1]:
                    depth_value = self.latest_depth[center_v, center_u]
                    if depth_value > 0 and np.isfinite(depth_value):
                        z_cam = depth_value
                        x_cam = (center_u - self.cx) * z_cam / self.fx
                        y_cam = (center_v - self.cy) * z_cam / self.fy

                        # Display the coordinates on the image
                        coord_text = f"x:{x_cam:.2f} y:{y_cam:.2f} z:{z_cam:.2f}"
                        cv2.putText(frame, coord_text, (center_u, center_v),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        self.get_logger().info(
                            f"Track {track_id} in camera_link: x={x_cam:.2f}, y={y_cam:.2f}, z={z_cam:.2f}"
                        )

                        # Only process and publish for person with id == 1
                        if int(track_id) == 1:
                            # Publish the transform using tf2
                            t = TransformStamped()
                            t.header.stamp = self.get_clock().now().to_msg()
                            t.header.frame_id = "camera_link"
                            t.child_frame_id = f"person_{track_id}"
                            # Map camera coordinates to map frame as needed
                            t.transform.translation.x = float(z_cam)
                            t.transform.translation.y = -(float(x_cam))
                            t.transform.translation.z = 0.0
                            # Use identity quaternion (no rotation)
                            t.transform.rotation.x = 0.0
                            t.transform.rotation.y = 0.0
                            t.transform.rotation.z = 0.0
                            t.transform.rotation.w = 1.0
                            self.tf_broadcaster.sendTransform(t)

                            try:
                                # Look up the transform from camera_link to map for the person.
                                transform = self.tf_buffer.lookup_transform(
                                    "map",
                                    f"person_{track_id}",
                                    self.get_clock().now().to_msg(),
                                    rclpy.duration.Duration(seconds=0.1)
                                )
                                self.get_logger().info(f"Transform for person_{track_id}: {transform}")

                                # Create a new transform for the goal pose.
                                # Parent frame is "map", and the goal is offset by 1 meter in the x direction.
                                goal_t = TransformStamped()
                                goal_t.header.stamp = self.get_clock().now().to_msg()
                                goal_t.header.frame_id = "map"
                                goal_t.child_frame_id = "goal_pose"
                                goal_t.transform.translation.x = transform.transform.translation.x - 0.3
                                goal_t.transform.translation.y = transform.transform.translation.y
                                goal_t.transform.translation.z = 0.0
                                goal_t.transform.rotation.x = 0.0
                                goal_t.transform.rotation.y = 0.0
                                goal_t.transform.rotation.z = 0.0
                                goal_t.transform.rotation.w = 1.0
                                self.tf_broadcaster.sendTransform(goal_t)

                            except Exception as e:
                                self.get_logger().warn(f"Transform from camera_link to map not available: {e}")

                            # try:
                            #     # Look up the transform from camera_link to map.
                            #     transform = self.tf_buffer.lookup_transform(
                            #         "map",
                            #         f"person_{track_id}",
                            #         rclpy.time.Time(),
                            #         rclpy.duration.Duration(seconds=0.1)
                            #     )
                            #     self.get_logger().info(f"Transform for person_{track_id}: {transform}")

                            #     # Create and publish goal pose with the z axis offset by 1 meter.
                            #     goal_pose = PoseStamped()
                            #     goal_pose.header.stamp = self.get_clock().now().to_msg()
                            #     goal_pose.header.frame_id = "map"
                            #     goal_pose.pose.position.x = t.transform.translation.x + 1.0
                            #     goal_pose.pose.position.y = -(t.transform.translation.y)
                            #     # Add 1 meter to the z axis
                            #     goal_pose.pose.position.z = 0.0
                            #     # Set identity orientation
                            #     goal_pose.pose.orientation.x = 0.0
                            #     goal_pose.pose.orientation.y = 0.0
                            #     goal_pose.pose.orientation.z = 0.0
                            #     goal_pose.pose.orientation.w = 1.0
                            #     self.goal_publisher.publish(goal_pose)
                            #     self.get_logger().info(f"Published goal pose for person_{track_id}: {goal_pose}")
                            # except Exception as e:
                            #     self.get_logger().warn(f"Transform from camera_link to map not available: {e}")
                    else:
                        self.get_logger().warn(f"Invalid depth value at ({center_u}, {center_v}).")
                else:
                    self.get_logger().warn("Center of bounding box is out of depth image bounds.")
            else:
                self.get_logger().warn("No depth image received yet.")

        # Display result in an OpenCV window
        cv2.imshow("YOLO + Deep SORT Tracking", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = RealsenseTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
