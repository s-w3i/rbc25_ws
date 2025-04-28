#!/usr/bin/env python3
"""Bag Pointing Centroid Service – camera0 (on-demand)

ROS 2 service **/get_bag_centroid**. When called, grabs the latest RGB + depth
frames, sends them to GPT-4o-vision with explicit image resolution context,
and expects a **normalized centroid** (`x_c`,`y_c`) or a `no_pointing` flag.
It then converts normalized coords to pixels, back-projects using depth + intrinsics,
and returns a metric 3-D centroid. Publishes a debug image overlaying the
centroid and optionally shows a live OpenCV window.

Service: robot_interfaces/srv/BagCentroid
----------------------------------------
```srv
# Request – empty
---
geometry_msgs/Point centroid  # XYZ in metres, camera optical frame
bool success                 # true if centroid returned
string message               # Info or error description
```
"""
from __future__ import annotations

import base64
import json
import os
from threading import Lock
from typing import Optional, Tuple

import cv2
import numpy as np
import openai
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import CameraInfo, Image

from robot_interfaces.srv import BagCentroid

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY", "")

class BagPointingCentroidService(Node):
    """Service node for on-demand bag-centroid detection with resolution-aware prompt."""

    def __init__(self) -> None:
        super().__init__("bag_pointing_centroid_service")
        qos = QoSPresetProfiles.SENSOR_DATA.value

        # Parameters
        self.declare_parameter("model", "gpt-4o-mini")
        self.declare_parameter("prompt", "")
        self.declare_parameter("show_window", True)
        self.declare_parameter("depth_patch_size", 5)
        self.declare_parameter("depth_scale", 0.001)

        self.model_name = self.get_parameter("model").value
        prm = self.get_parameter("prompt").value
        self.custom_prompt = prm if prm else None
        self.show_window = self.get_parameter("show_window").value
        self.depth_patch_size = int(self.get_parameter("depth_patch_size").value)
        self.depth_scale = float(self.get_parameter("depth_scale").value)

        # State
        self.bridge = CvBridge()
        self.rgb_image: Optional[np.ndarray] = None
        self.depth_image: Optional[np.ndarray] = None
        self.fx = self.fy = self.cx = self.cy = 0.0
        self.intrinsics_ready = False
        self.window_created = False
        self.img_lock = Lock()

        # Subscriptions
        self.create_subscription(Image, "/camera0/color/image_raw", self.rgb_cb, qos)
        self.create_subscription(Image, "/camera0/depth/image_rect_raw", self.depth_cb, qos)
        self.create_subscription(CameraInfo, "/camera0/color/camera_info", self.info_cb, qos)

        # Publisher & Service
        self.debug_pub = self.create_publisher(Image, "/bag_detection_debug", 10)
        self.create_service(BagCentroid, "get_bag_centroid", self.handle_request)

        if not openai.api_key:
            self.get_logger().error("OPENAI_API_KEY not set – service disabled.")
        self.get_logger().info("Service /get_bag_centroid ready.")

    def info_cb(self, msg: CameraInfo) -> None:
        if not self.intrinsics_ready:
            self.fx, self.fy, self.cx, self.cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
            self.intrinsics_ready = True
            self.get_logger().info("Camera intrinsics acquired.")

    def rgb_cb(self, msg: Image) -> None:
        with self.img_lock:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def depth_cb(self, msg: Image) -> None:
        with self.img_lock:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def handle_request(self, _req: BagCentroid.Request, resp: BagCentroid.Response):
        # Ensure data readiness
        if not self.intrinsics_ready:
            resp.success = False
            resp.message = "Camera intrinsics not yet received."
            return resp
        with self.img_lock:
            rgb = self.rgb_image.copy() if self.rgb_image is not None else None
            depth = self.depth_image.copy() if self.depth_image is not None else None
        if rgb is None or depth is None:
            resp.success = False
            resp.message = "No image data available."
            return resp

        h, w = rgb.shape[:2]
        # Build prompt including resolution
        default_prompt = (
            f"Image resolution: {w}×{h} pixels. There may be a person pointing at a bag. "
            "If a bag is pointed at, return ONLY JSON {\"x_c\":float,\"y_c\":float} "
            "with centroid normalized to [0,1]. If no pointing, return JSON {\"no_pointing\":true}."
        )
        prompt_text = self.custom_prompt or default_prompt

        # Encode image to base64
        _, buf = cv2.imencode('.jpg', rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        b64 = base64.b64encode(buf).decode()

        # Call GPT-4o Vision
        try:
            completion = openai.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }],
                max_tokens=60,
                temperature=0.0,
            )
            raw = completion.choices[0].message.content.strip()
            if not raw:
                resp.success = False
                resp.message = "Empty response from GPT-4o."
                return resp
            data = json.loads(raw)
            if isinstance(data, dict) and data.get("no_pointing"):
                resp.success = False
                resp.message = "No pointing gesture detected."
                return resp
            x_c = float(data["x_c"])
            y_c = float(data["y_c"])
        except Exception as exc:
            resp.success = False
            resp.message = f"GPT-4o error: {exc}"
            return resp

        # Pixel centroid
        u = int(np.clip(x_c * w, 0, w-1))
        v = int(np.clip(y_c * h, 0, h-1))

        # Depth patch median
        half = self.depth_patch_size // 2
        patch = depth[max(0, v-half):v+half+1, max(0, u-half):u+half+1]
        if patch.dtype == np.uint16:
            patch = patch.astype(np.float32) * self.depth_scale
        valid = patch[(patch>0) & ~np.isnan(patch)]
        if valid.size == 0:
            resp.success = False
            resp.message = "No valid depth at centroid pixel."
            return resp
        z_med = float(np.median(valid))

        # Back-project
        X = (u - self.cx) * z_med / self.fx
        Y = (v - self.cy) * z_med / self.fy
        Z = z_med
        resp.centroid = Point(x=X, y=Y, z=Z)
        resp.success = True
        resp.message = "Centroid calculated successfully."

        # Debug overlay
        debug = rgb.copy()
        cv2.circle(debug, (u, v), 5, (0, 0, 255), -1)
        cv2.putText(debug, f"{X:.2f}, {Y:.2f}, {Z:.2f} m", (u+10, v-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding="bgr8"))
        if self.show_window and not self.window_created:
            cv2.namedWindow("Bag Detection Result", cv2.WINDOW_NORMAL)
            self.window_created = True
        if self.show_window:
            cv2.imshow("Bag Detection Result", debug)
            cv2.waitKey(1)

        return resp


def main(args=None):
    rclpy.init(args=args)
    node = BagPointingCentroidService()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
