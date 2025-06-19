#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Vector3, PoseStamped, TransformStamped
from rclpy.qos import QoSPresetProfiles
from cv_bridge import CvBridge
import cv2
import numpy as np
import subprocess
from std_srvs.srv import Trigger
from ultralytics import YOLO
from robot_interfaces.msg import Detections, BoundingBox
from builtin_interfaces.msg import Time as TimeMsg
import tf2_ros
import tf2_geometry_msgs
import os

class YOLOEmptyChairService(Node):
    def __init__(self):
        super().__init__('yolo_empty_chair_service')
        self.bridge = CvBridge()

        # QoS
        qos_sensor_data = QoSPresetProfiles.SENSOR_DATA.value
        self.save_dir = os.path.expanduser('~/empty_chair_detections')
        os.makedirs(self.save_dir, exist_ok=True)


        # Subscribers
        self.color_sub = self.create_subscription(Image, '/camera0/color/image_raw', self.color_callback, qos_sensor_data)
        self.depth_sub = self.create_subscription(Image, '/camera0/depth/image_rect_raw', self.depth_callback, qos_sensor_data)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera0/color/camera_info', self.camera_info_callback, qos_sensor_data)

        # Service
        self.srv = self.create_service(Trigger, 'detect_empty_chair', self.service_callback)

        # Publishers
        self.detection_pub = self.create_publisher(Detections, '/empty_chair_detections', 10)
        self.pose_pub_raw = self.create_publisher(PoseStamped, '/chair_pose_raw', 10)
        self.tf_pub = self.create_publisher(TransformStamped, '/chair_tf', 10)
        self.arm_target_pub = self.create_publisher(Vector3, '/arm_target', 10)

        # TF
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # YOLO model
        self.yolo = YOLO('yolo11s-seg.pt')
        self.get_logger().info(f"YOLO segmentation model loaded. Names: {self.yolo.names}")

        # Data storage
        self.latest_color = None
        self.latest_depth = None
        self.latest_camera_info = None

        # frames
        self.base_frame = 'base_link'
        self.sensor_frame = 'camera0_color_optical_frame'
        # arm reach
        self.arm_reach = 0.3

    def calculate_iou(self, boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        interW, interH = max(0, xB-xA), max(0, yB-yA)
        interArea = interW * interH
        if interArea == 0: return 0.0
        boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
        boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
        return interArea/(boxAArea + boxBArea - interArea)

    def color_callback(self, msg):
        try: self.latest_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e: self.get_logger().error(f"Error converting color image: {e}")

    def depth_callback(self, msg):
        try: self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e: self.get_logger().error(f"Error converting depth image: {e}")

    def camera_info_callback(self, msg):
        self.latest_camera_info = msg

    def service_callback(self, request, response):
        now = self.get_clock().now().to_msg()
        # check data
        if self.latest_color is None or self.latest_depth is None or self.latest_camera_info is None:
            response.success = False
            response.message = "Waiting for color, depth, and camera info"
            return response

        cv_image = self.latest_color.copy()
        depth_image = self.latest_depth
        cam_info = self.latest_camera_info

        # YOLO inference
        results = self.yolo(cv_image)
        res = results[0]

        # detections
        chairs, persons = [], []
        for box, conf, cls in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
            if conf < 0.5: continue
            x1,y1,x2,y2 = map(int, box.tolist())
            label = self.yolo.names[int(cls)].lower()
            center = ((x1+x2)//2, (y1+y2)//2)
            entry = {'bbox':[x1,y1,x2,y2], 'conf':float(conf), 'center':center}
            (chairs if label=='chair' else persons).append(entry)

        empty = [c for c in chairs if not any(self.calculate_iou(c['bbox'],p['bbox'])>0.2 for p in persons)]

        # find nearest
        nearest,min_d = None, float('inf')
        for c in empty:
            cx,cy = c['center']
            z = depth_image[cy,cx]/1000.0
            if z<=0: continue
            fx,fy = cam_info.k[0], cam_info.k[4]
            cxi,cyi = cam_info.k[2], cam_info.k[5]
            X = (cx-cxi)*z/fx
            Y = (cy-cyi)*z/fy
            d = np.linalg.norm([X,Y,z])
            if d<min_d: min_d, nearest=(d,(c,(X,Y,z)))

        det_msg = Detections()
        if nearest:
            chair,(Xc,Yc,zc) = nearest
            bbox = chair['bbox']
            b = BoundingBox(x1=bbox[0],y1=bbox[1],x2=bbox[2],y2=bbox[3],confidence=chair['conf'],distance=min_d)
            cv2.rectangle(cv_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
            text = f"x:{Xc:.2f}m y:{Yc:.2f}m z:{zc:.2f}m"
            cv2.putText(cv_image, text, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            det_msg.detections.append(b)
            ts = now.sec
            ns = now.nanosec // 1_000_000  # milliseconds
            fname = f"empty_chair_{ts}_{ns:03d}.png"
            fpath = os.path.join(self.save_dir, fname)
            cv2.imwrite(fpath, cv_image)
            # raw pose
            pose_raw = PoseStamped()
            pose_raw.header.stamp = now
            pose_raw.header.frame_id = self.sensor_frame
            pose_raw.pose.position.x = Xc
            pose_raw.pose.position.y = Yc
            pose_raw.pose.position.z = zc
            pose_raw.pose.orientation.w = 1.0
            self.pose_pub_raw.publish(pose_raw)
            # transform
            pose_tf = PoseStamped()
            pose_tf.header = pose_raw.header
            pose_tf.header.stamp = TimeMsg()
            pose_tf.pose = pose_raw.pose
            try:
                pout = self.tf_buffer.transform(pose_tf, self.base_frame)
                pout.pose.position.z = 0.0
                # broadcast
                tf_msg = TransformStamped()
                tf_msg.header.stamp = now
                tf_msg.header.frame_id = self.base_frame
                tf_msg.child_frame_id = 'empty_chair'
                tf_msg.transform.translation.x = pout.pose.position.x
                tf_msg.transform.translation.y = pout.pose.position.y
                tf_msg.transform.translation.z = pout.pose.position.z
                q = pout.pose.orientation
                tf_msg.transform.rotation.x = q.x
                tf_msg.transform.rotation.y = q.y
                tf_msg.transform.rotation.z = q.z
                tf_msg.transform.rotation.w = q.w
                self.tf_broadcaster.sendTransform(tf_msg)
                self.tf_pub.publish(tf_msg)
                # compute arm target
                xb, yb = pout.pose.position.x, pout.pose.position.y
                R = self.arm_reach
                d_xy = np.hypot(xb,yb)
                if d_xy>R:
                    scale = R/d_xy
                    xt, yt = xb*scale, yb*scale
                else:
                    xt, yt = xb, yb 
                # publish arm target
                v = Vector3()
                v.x, v.y, v.z = xt, yt, 0.22
                self.arm_target_pub.publish(v)
                # response
                response.success = True
                response.message = f"Arm target: x:{xt:.3f} y:{yt:.3f}"
            except Exception as ex:
                self.get_logger().warn(f"TF error: {ex}")
                response.success=False
                response.message="TF transform failed"
        else:
            response.success = False
            response.message = "No empty chair detected"

        self.detection_pub.publish(det_msg)

        # display window
        win = "Empty Chair"
        h,w = cv_image.shape[:2]
        cv2.namedWindow(win,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win,w,h)
        cv2.imshow(win,cv_image)
        cv2.waitKey(1)
        try:
            subprocess.run(["wmctrl","-r",win,"-b","add,above"],check=True)
            subprocess.run(["wmctrl","-a",win],check=True)
        except:
            pass
        cv2.waitKey(5000)
        cv2.destroyWindow(win)

        return response


def main(args=None):
    rclpy.init(args=args)
    node = YOLOEmptyChairService()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()