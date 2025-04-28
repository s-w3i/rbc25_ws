#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.qos import QoSPresetProfiles, QoSProfile
from rclpy.qos import DurabilityPolicy
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_srvs.srv import Trigger
from ultralytics import YOLO
from robot_interfaces.msg import Detections, BoundingBox
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from rclpy.duration import Duration
from builtin_interfaces.msg import Time as TimeMsg

class YOLOEmptyChairService(Node):
    def __init__(self):
        super().__init__('yolo_empty_chair_service')
        self.bridge = CvBridge()

        # QoS for sensor data
        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value

        # Subscribers
        self.create_subscription(Image, '/camera0/color/image_raw', self.color_callback, sensor_qos)
        self.create_subscription(Image, '/camera0/depth/image_rect_raw', self.depth_callback, sensor_qos)
        self.create_subscription(CameraInfo, '/camera0/color/camera_info', self.camera_info_callback, sensor_qos)

        # TF: buffer, listener, broadcaster
        self.tf_buffer = Buffer(cache_time=Duration(seconds=60.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Publisher for chair transform
        qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.chair_transform_pub = self.create_publisher(TransformStamped, 'chair_transform', qos)

        # Pose and detection publishers
        self.pose_pub_raw = self.create_publisher(PoseStamped, '/chair_pose_raw', 10)
        self.detection_pub = self.create_publisher(Detections, '/empty_chair_detections', 10)

        # Service
        self.create_service(Trigger, 'detect_empty_chair', self.service_callback)

        # Load YOLO segmentation model
        self.yolo = YOLO('yolo11s-seg.pt')
        self.get_logger().info(f"YOLO segmentation model loaded. Names: {self.yolo.names}")

        # Data holders
        self.latest_color = None
        self.latest_depth = None
        self.latest_camera_info = None
        self.sensor_frame = 'camera0_color_optical_frame'

    def calculate_iou(self, boxA, boxB):
        # unchanged
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        interW, interH = max(0, xB - xA), max(0, yB - yA)
        interArea = interW * interH
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / (boxAArea + boxBArea - interArea)

    def color_callback(self, msg: Image):
        # unchanged
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error converting color image: {e}")

    def depth_callback(self, msg: Image):
        # unchanged
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) / 1000.0
            self.latest_depth = depth
        except Exception as e:
            self.get_logger().error(f"Depth conversion error: {e}")

    def camera_info_callback(self, msg: CameraInfo):
        # unchanged
        self.latest_camera_info = msg

    def service_callback(self, request, response):
        now = self.get_clock().now().to_msg()

        # validate sensor inputs
        if self.latest_color is None or self.latest_depth is None or self.latest_camera_info is None:
            response.success = False
            response.message = "Waiting for color, depth, and camera info"
            return response

        cv_image = self.latest_color.copy()
        depth = self.latest_depth
        cam_info = self.latest_camera_info

        # YOLO segmentation
        results = self.yolo(cv_image)
        res = results[0]

        # separate chairs and persons
        chairs, persons = [], []
        for box, conf, cls in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
            if conf < 0.5:
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            label = self.yolo.names[int(cls)].lower()
            center = ((x1 + x2)//2, (y1 + y2)//2)
            entry = {'bbox': [x1, y1, x2, y2], 'conf': float(conf), 'center': center}
            (chairs if label == 'chair' else persons).append(entry)

        # filter empty chairs
        empty = [c for c in chairs if not any(
            self.calculate_iou(c['bbox'], p['bbox']) > 0.2 for p in persons
        )]

        # find nearest empty chair
        nearest, min_d = None, float('inf')
        for c in empty:
            cx, cy = c['center']
            z = depth[cy, cx]
            if z <= 0:
                continue
            fx, fy = cam_info.k[0], cam_info.k[4]
            cxi, cyi = cam_info.k[2], cam_info.k[5]
            X = (cx - cxi) * z / fx
            Y = (cy - cyi) * z / fy
            d = np.linalg.norm([X, Y, z])
            if d < min_d:
                min_d, nearest = d, (c, (X, Y, z))

        det_msg = Detections()
        if nearest:
            chair, (Xc, Yc, zc) = nearest
            bbox = chair['bbox']
            b = BoundingBox(
                x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3],
                confidence=chair['conf'], distance=min_d
            )
            det_msg.detections.append(b)

            # raw pose
            pose_raw = PoseStamped()
            pose_raw.header.stamp = now
            pose_raw.header.frame_id = self.sensor_frame
            pose_raw.pose.position.x = Xc
            pose_raw.pose.position.y = Yc
            pose_raw.pose.position.z = 0.0
            pose_raw.pose.orientation.w = 1.0
            self.pose_pub_raw.publish(pose_raw)

            # transform to map frame
            pose_tf = PoseStamped()
            pose_tf.header.stamp = TimeMsg()
            pose_tf.header.frame_id = pose_raw.header.frame_id
            pose_tf.pose = pose_raw.pose
            try:
                map_pose = self.tf_buffer.transform(pose_tf, 'map')
                map_pose.pose.position.z = 0.0  # Flatten Z
                tf_msg = TransformStamped()
                tf_msg.header.stamp = now
                tf_msg.header.frame_id = 'map'
                tf_msg.child_frame_id = 'empty_chair'
                tf_msg.transform.translation.x = map_pose.pose.position.x
                tf_msg.transform.translation.y = map_pose.pose.position.y
                tf_msg.transform.translation.z = map_pose.pose.position.z
                tf_msg.transform.rotation = map_pose.pose.orientation

                # broadcast via TF2
                self.tf_broadcaster.sendTransform(tf_msg)
                # publish as a topic
                self.chair_transform_pub.publish(tf_msg)
            except Exception as e:
                self.get_logger().warn(f"TF transform failed for empty_chair: {e}")

            # publish detection message
            self.detection_pub.publish(det_msg)

            response.success = True
            response.message = "Empty chair detected and published"
        else:
            response.success = False
            response.message = "No empty chair found"

        return response


def main(args=None):
    rclpy.init(args=args)
    node = YOLOEmptyChairService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
