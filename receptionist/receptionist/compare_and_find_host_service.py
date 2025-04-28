#!/usr/bin/env python3
import os
import glob
import json
import base64
import cv2

from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point

from openai import OpenAI
from robot_interfaces.srv import CompareAndFindHost  # adjust to your pkg name


class CompareAndFindHostService(Node):
    def __init__(self):
        super().__init__('compare_and_find_host')
        self.bridge = CvBridge()
        self.openai = OpenAI()
        self.srv = self.create_service(
            CompareAndFindHost,
            'compare_and_find_host',
            self.callback
        )
        self.get_logger().info('Service compare_and_find_host ready.')

    def _load_first_file(self, directory: str) -> str:
        imgs = []
        for ext in ('jpg','jpeg','png'):
            imgs += glob.glob(os.path.join(directory, f'*.{ext}'))
        imgs.sort()
        if not imgs:
            raise FileNotFoundError(f"No images in {directory}")
        return imgs[0]

    def _encode_file(self, path: str) -> str:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        _, buf = cv2.imencode('.jpg', img)
        return f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"

    def _encode_msg(self, img_msg: Image) -> str:
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        _, buf = cv2.imencode('.jpg', cv_img)
        return f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"

    def callback(self, req, resp):
        try:
            # 1) pick host file
            host_path = req.host_image_dir
            if os.path.isdir(host_path):
                host_path = self._load_first_file(host_path)
            if not os.path.isfile(host_path):
                raise FileNotFoundError(f"Host not found: {host_path}")

            # 2) encode host + segments
            host_uri = self._encode_file(host_path)
            seg_uris = [self._encode_msg(im) for im in req.segmented_images]

            # 3) build prompt asking for JSON { "best_index": N }
            listing = "\n".join(f"[{i}] – centroid=({c.x:.1f},{c.y:.1f})"
                                 for i, c in enumerate(req.centroids))
            intro = (
                "You have one host image, followed by several segmented images whose pixel centroids are listed below:\n"
                f"{listing}\n\n"
                "Exactly one segmented image shows the same person as the host.  \n"
                "Return JSON exactly like: { \"best_index\": N }  \n"
                "—where best_index is the 0-based index of the matching segment."
            )

            content = [{"type":"text","text":intro}]
            content.append({"type":"image_url","image_url":{"url":host_uri}})
            for uri in seg_uris:
                content.append({"type":"image_url","image_url":{"url":uri}})

            # 4) call the OpenAI Vision endpoint
            r = self.openai.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role":"user","content":content}],
            )
            reply = r.choices[0].message.content.strip()

            # 5) parse JSON
            parsed = json.loads(reply)
            idx = parsed["best_index"]
            if not (0 <= idx < len(req.segmented_images)):
                raise IndexError(f"best_index {idx} out of range")

            # 6) return person_<1-based>
            resp.success    = True
            resp.best_person = f"person_{idx+1}"

        except Exception as e:
            self.get_logger().error(f"compare error: {e}")
            resp.success     = False
            resp.best_person = ""

        return resp


def main(args=None):
    rclpy.init(args=args)
    node = CompareAndFindHostService()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
