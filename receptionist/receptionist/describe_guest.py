#!/usr/bin/env python3
import os
import json
import base64
import asyncio

import rclpy
from rclpy.node import Node
from openai import AsyncOpenAI

from robot_interfaces.srv import DescribeGuest  # Ensure this matches your .srv

class GuestDescriptionNode(Node):
    def __init__(self):
        super().__init__('guest_description_node')
        self.srv = self.create_service(
            DescribeGuest,
            'describe_guest',
            self.describe_guest_callback
        )

        self.ai_client = AsyncOpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        self.model = "gpt-4o-mini"  # vision-capable model

        if not self.ai_client.api_key:
            self.get_logger().error("OPENAI_API_KEY not set in environment.")
        else:
            self.get_logger().info("OpenAI key found, ready to describe guests.")

    def encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        b64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"

    async def async_describe_person(self, image_path: str) -> str:
        data_url = self.encode_image_to_base64(image_path)

        # 1) System prompt: enforce numeric height_cm and age keys
        system_msg = {
            "role": "system",
            "content": (
                "You are an image analysis assistant. "
                "When given an image, you must output only a single JSON object "
                "with exactly these keys:\n"
                "  • gender     (\"male\" or \"female\")\n"
                "  • hair_colour\n"
                "  • wearing_hat (true/false)\n"
                "  • color_of_pants\n"
                "  • wearing_glasses (true/false)\n"
                "  • wearing_mask   (true/false)\n"
                "  • color_of_clothes\n"
                "Do not wrap in markdown or add commentary—just raw JSON."
            )
        }

        # 2) User message with the image
        user_msg = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Analyze the image and return one JSON object with the keys "
                        "gender, hair_colour, wearing_hat, color_of_pants, "
                        "wearing_glasses, wearing_mask, color_of_clothes."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_url}
                }
            ]
        }

        # 3) Call the API
        resp = await self.ai_client.chat.completions.create(
            model=self.model,
            messages=[system_msg, user_msg]
        )
        raw = resp.choices[0].message.content.strip()

        # 4) Strip code fences if needed
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(lines[1:-1])

        # 5) Validate JSON
        parsed = json.loads(raw)

        # 6) Re‑dump for stability
        return json.dumps(parsed, ensure_ascii=False)

    def describe_person(self, image_path: str) -> str:
        try:
            return asyncio.run(self.async_describe_person(image_path))
        except Exception as e:
            self.get_logger().error(f"OpenAI call failed: {e}")
            raise

    def describe_guest_callback(self, request, response):
        self.get_logger().info(f"Received describe_guest request for {request.image_path}")
        try:
            description = self.describe_person(request.image_path)
            response.success = True
            response.description = description
            response.message = "Guest described successfully."
            self.get_logger().info(f"Description: {description}")
        except Exception as e:
            response.success = False
            response.description = ""
            response.message = str(e)
            self.get_logger().error(response.message)
        return response

def main(args=None):
    rclpy.init(args=args)
    node = GuestDescriptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
