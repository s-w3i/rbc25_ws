#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_srvs.srv import SetBool
from builtin_interfaces.msg import Time as TimeMsg    # zero-stamp for TF lookup

from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

import tf2_ros
import tf2_geometry_msgs                                # registers PoseStamped with TF2


class LiveHumanDetectionWithID(Node):
    """YOLOv8-seg + DeepSORT + adaptive Re-ID for ROS 2."""

    # ──────────── init ────────────
    def __init__(self):
        super().__init__("live_human_detection_with_service")
        self.bridge = CvBridge()

        # TF / frame ids
        self.sensor_frame = "camera0_color_optical_frame"
        self.base_frame   = "map"

        self.tf_buffer      = tf2_ros.Buffer()
        self.tf_listener    = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # pubs
        self.pose_pub_raw = self.create_publisher(PoseStamped, "/people_pose_raw", 10)
        self.tf_pub       = self.create_publisher(TransformStamped, "/people_transform", 10)

        # toggle service
        self.active = False
        self.srv = self.create_service(SetBool, "toggle_detection", self.toggle_detection_cb)
        self.get_logger().info("Service '/toggle_detection' ready (SetBool).")

        # subs
        qos = QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(Image,      "/camera0/color/image_raw",      self.color_cb, qos)
        self.create_subscription(Image,      "/camera0/depth/image_rect_raw", self.depth_cb,  qos)
        self.create_subscription(CameraInfo, "/camera0/color/camera_info",    self.info_cb,   qos)

        # detector  – segmentation model (gives person masks)
        self.model = YOLO("yolov8n-seg.pt")                 # download once
        self.get_logger().info("Loaded YOLOv8-seg model.")

        # DeepSORT
        self.tracker = DeepSort(
            max_age=5000,              # frames to keep "lost" track alive
            n_init=15,                 # hits to confirm a track
            max_iou_distance=0.3,
            nms_max_overlap=0.8,
            embedder="mobilenet",      # keep CPU-light; can swap to torchreid later
            embedder_gpu=torch.cuda.is_available()
        )

        # buffers
        self.latest_color = None
        self.latest_depth = None
        self.fx = self.fy = self.cx = self.cy = None

        # Re-ID memory
        self.gallery     = {}          # id → feature vector
        self.last_seen   = {}          # id → rclpy.time.Time
        self.base_thresh = 0.65        # adaptive lower bound

        # run at 10 Hz
        self.create_timer(0.1, self.process_frame)

    # ──────────── callbacks ────────────
    def toggle_detection_cb(self, req, res):
        self.active = req.data
        res.success = True
        res.message = f"Detection {'enabled' if self.active else 'disabled'}"
        self.get_logger().info(res.message)
        return res

    def color_cb(self, msg: Image):
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Color conversion error: {e}")

    def depth_cb(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            if depth.dtype == np.uint16:          # mm → m
                depth = depth.astype(np.float32) / 1000.0
            self.latest_depth = depth
        except Exception as e:
            self.get_logger().error(f"Depth conversion error: {e}")

    def info_cb(self, msg: CameraInfo):
        K = msg.k
        self.fx, self.fy = K[0], K[4]
        self.cx, self.cy = K[2], K[5]

    # ──────────── main loop ────────────
    def process_frame(self):
        if not self.active:
            return
        if self.latest_color is None or self.latest_depth is None or self.fx is None:
            return

        frame = self.latest_color.copy()

        # 1 · Detect persons + masks
        results   = self.model(frame, conf=0.5, classes=[0])[0]     # person class id = 0
        boxes_xy  = results.boxes.xyxy.cpu().numpy().astype(int) if results.boxes is not None else []
        confs     = results.boxes.conf.cpu().numpy() if results.boxes is not None else []
        masks_all = results.masks.data.cpu().numpy() if results.masks is not None else None

        # 2 · Build DeepSORT dets + masks
        dets, instance_masks = [], []
        for i, ((x1, y1, x2, y2), c) in enumerate(zip(boxes_xy, confs)):
            dets.append([[x1, y1, x2 - x1, y2 - y1], float(c), 0])
            instance_masks.append((masks_all[i] > 0.5) if masks_all is not None else None)

        # 3 · Preserve current gallery
        old_gallery = self.gallery.copy()

        # 4 · Update tracks (Option 4 – pass masks)
        tracks = self.tracker.update_tracks(dets, frame=frame, instance_masks=instance_masks)

        # 5 · Refresh embeddings for confirmed tracks
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            tid, feat = tr.track_id, tr.get_feature()
            if feat is not None:
                self.gallery[tid] = feat / np.linalg.norm(feat)

        # 6 · Re-associate new IDs  (Option 3 – adaptive threshold)
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            tid = tr.track_id
            if tid in old_gallery:
                continue                          # already known

            new_feat = tr.get_feature()
            if new_feat is None:
                continue
            new_feat /= np.linalg.norm(new_feat)

            # best cosine similarity against gallery
            best_id, best_sim = None, -1.0
            for gid, gfeat in old_gallery.items():
                sim = float(np.dot(new_feat, gfeat))
                if sim > best_sim:
                    best_sim, best_id = sim, gid

            # adaptive threshold depends on how long best_id has been gone
            if best_id is not None and best_id in self.last_seen:
                gap_s = (self.get_clock().now() - self.last_seen[best_id]).nanoseconds * 1e-9
            else:
                gap_s = 0.0
            thr = np.interp(gap_s, [0, 10, 30], [0.75, self.base_thresh, self.base_thresh - 0.1])

            if best_sim > thr:
                self.get_logger().info(f"Re-ID: {tid} → {best_id} (sim={best_sim:.2f}, thr={thr:.2f})")
                tr.track_id          = best_id   # replace id in tracker
                self.gallery[best_id] = new_feat # update appearance

        # 7 · Publish / draw
        now_msg = self.get_clock().now().to_msg()
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            tid      = tr.track_id
            self.last_seen[tid] = self.get_clock().now()          # update seen time

            l, t, r, b = map(int, tr.to_ltrb())
            cx, cy     = (l + r) // 2, (t + b) // 2
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            if 0 <= cx < self.latest_depth.shape[1] and 0 <= cy < self.latest_depth.shape[0]:
                z = float(self.latest_depth[cy, cx])
                if z > 0.0 and np.isfinite(z):
                    X = (cx - self.cx) * z / self.fx
                    Y = (cy - self.cy) * z / self.fy

                    # camera-frame pose
                    pose_raw = PoseStamped()
                    pose_raw.header.stamp     = now_msg
                    pose_raw.header.frame_id  = self.sensor_frame
                    pose_raw.pose.position.x  = X
                    pose_raw.pose.position.y  = Y
                    pose_raw.pose.position.z  = z
                    pose_raw.pose.orientation.w = 1.0
                    self.pose_pub_raw.publish(pose_raw)

                    # transform to base frame
                    pose_tf           = pose_raw
                    pose_tf.header.stamp = TimeMsg()              # zero for latest
                    try:
                        pout = self.tf_buffer.transform(pose_tf, self.base_frame)
                        pout.pose.position.z = 0.0               # flatten

                        tf_msg                       = TransformStamped()
                        tf_msg.header.stamp          = now_msg
                        tf_msg.header.frame_id       = self.base_frame
                        tf_msg.child_frame_id        = f"people_{tid}"
                        tf_msg.transform.translation.x = pout.pose.position.x
                        tf_msg.transform.translation.y = pout.pose.position.y
                        tf_msg.transform.translation.z = pout.pose.position.z
                        tf_msg.transform.rotation    = pout.pose.orientation
                        self.tf_broadcaster.sendTransform(tf_msg)
                        self.tf_pub.publish(tf_msg)
                    except tf2_ros.TransformException as ex:
                        self.get_logger().warn(f"TF failure: {ex}")

                    cv2.putText(frame, f"ID {tid}: {z:.2f} m", (l, t - 8),
                                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Live Re-ID", frame)
        cv2.waitKey(1)


# ──────────── main ────────────
def main():
    rclpy.init()
    node = LiveHumanDetectionWithID()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
