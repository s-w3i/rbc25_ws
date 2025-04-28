#!/usr/bin/env python3
import os
import base64
import cv2
import json
import requests
import numpy as np
from io import BytesIO

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge

from openai import OpenAI

from open_task.image_checker import ImageChecker
from open_task.voice_classifier import AssistantClassifier


class AssistantNode(Node):
    def __init__(self):
        super().__init__('assistant_node')

        # Read API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self.get_logger().error("OPENAI_API_KEY not set in environment!")
            raise RuntimeError("Missing OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

        # Helpers
        self.bridge = CvBridge()
        self.latest_frame = None

        # Subscriptions
        self.create_subscription(
            RosImage,
            '/camera0/color/image_raw',         # RealSense color frames
            self.camera_callback,
            10
        )
        self.create_subscription(
            String,
            '/speech_recognition_transcript',  # ASR → text
            self.speech_callback,
            10
        )

        # Publishers
        self.text_pub = self.create_publisher(String, '/assistant/response_text', 10)
        self.img_pub  = self.create_publisher(String, '/assistant/response_image_path', 10)

        # Your helper classes (they also read OPENAI_API_KEY themselves)
        self.img_checker = ImageChecker()
        self.voice_cls   = AssistantClassifier()

        self.get_logger().info("AssistantNode ready; listening for transcripts and camera frames.")

    def camera_callback(self, msg: RosImage):
        """Keep the latest RealSense color frame."""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"Image conversion failed: {e}")

    def speech_callback(self, msg: String):
        """Process each speech‐to‐text transcript and publish a response."""
        text = msg.data.strip()
        self.get_logger().info(f"Received transcript: {text!r}")
        reply_text, image_path = self.process_request(text)

        # Publish text reply
        out_txt = String()
        out_txt.data = reply_text
        self.text_pub.publish(out_txt)

        # If image_path is non-empty (analysis), publish it
        if image_path:
            out_img = String()
            out_img.data = image_path
            self.img_pub.publish(out_img)

    def encode_image(self, cv_img):
        """Encode BGR OpenCV image to base64 JPEG."""
        _, buf = cv2.imencode('.jpg', cv_img)
        return base64.b64encode(buf.tobytes()).decode('utf-8')

    def process_request(self, text: str):
        """
        Decide: image analysis, image generation, or chat.
        Returns (response_text, image_path_or_empty).
        """
        # 1) Image analysis?
        chk = self.img_checker.classify(text)
        if chk.get("type") == "yes" and self.latest_frame is not None:
            img_path = os.path.expanduser("~/analysis.jpg")
            cv2.imwrite(img_path, self.latest_frame)

            b64 = self.encode_image(self.latest_frame)
            ai_resp = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": chk.get("reason", "")},
                    {"role": "user",   "content": text},
                    {"role": "user",   "content": {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    }}
                ]
            ).choices[0].message.content

            return ai_resp, img_path

        # 2) Image generation?  → show on screen for 10s
        if any(k in text.lower() for k in ("draw", "generate", "show me", "picture of")):
            gen = self.client.images.generate(
                model="dall-e-3",
                prompt=text,
                size="1024x1024",
                n=1
            )
            url = gen.data[0].url
            resp = requests.get(url)
            # decode into OpenCV image
            arr = np.frombuffer(resp.content, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                cv2.imshow("Generated Image", img)
                cv2.waitKey(10000)   # display for 10 seconds
                cv2.destroyWindow("Generated Image")
                return "Displayed generated image for 10 seconds.", ""
            else:
                return "Failed to decode generated image.", ""

        # 3) Fallback: chat
        chat = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": text}],
            temperature=0
        )
        return chat.choices[0].message.content, ""


def main(args=None):
    rclpy.init(args=args)
    node = AssistantNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
