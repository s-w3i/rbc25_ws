#!/usr/bin/env python3
import os
import json
import asyncio

import rclpy
from rclpy.node import Node
from openai import AsyncOpenAI

from robot_interfaces.srv import DescribeGuestSentence

class GuestSentenceNode(Node):
    def __init__(self):
        super().__init__('guest_sentence_node')
        self.srv = self.create_service(
            DescribeGuestSentence,
            'describe_guest_sentence',
            self.describe_sentence_callback
        )

        self.ai_client = AsyncOpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        self.model = "gpt-4"

        if not self.ai_client.api_key:
            self.get_logger().error("OPENAI_API_KEY not set in environment.")
        else:
            self.get_logger().info("OpenAI key found, ready to generate guest sentences.")

    async def async_generate_sentence(self, guest_info_json: str) -> str:
        info = json.loads(guest_info_json)

        system_msg = {
            "role": "system",
            "content": (
                "You are a friendly assistant that turns structured guest information "
                "into a single natural-sounding English sentence. "
                "Only mention attributes whose value is not false; omit any boolean fields set to false."
            )
        }
        user_msg = {
            "role": "user",
            "content": (
                f"Here is the guest info:\n{json.dumps(info, ensure_ascii=False)}\n\n"
                "Write one descriptive sentence."
            )
        }

        resp = await self.ai_client.chat.completions.create(
            model=self.model,
            messages=[system_msg, user_msg]
        )
        return resp.choices[0].message.content.strip()

    def generate_sentence(self, guest_info_json: str) -> str:
        try:
            return asyncio.run(self.async_generate_sentence(guest_info_json))
        except Exception as e:
            self.get_logger().error(f"OpenAI call failed: {e}")
            raise

    def describe_sentence_callback(self, request, response):
        # 1) Print/log raw incoming JSON
        raw = request.guest_info_json
        self.get_logger().info(f"Raw guest_info_json: {raw}")
        print(">> Raw guest_info_json:\n", raw)

        # 2) Load and clean it
        info = json.loads(raw)
        # Remove unwanted fields
        info.pop("segmented_file", None)
        info.pop("centroids", None)
        # Remove any boolean fields that are False
        filtered = {
            k: v for k, v in info.items()
            if not (isinstance(v, bool) and v is False)
        }

        cleaned_json = json.dumps(filtered, ensure_ascii=False)
        self.get_logger().info(f"Cleaned guest_info_json: {cleaned_json}")
        print(">> Cleaned guest_info_json:\n", cleaned_json)

        # 3) Generate the sentence
        try:
            sentence = self.generate_sentence(cleaned_json)
            response.success = True
            response.description = sentence
            response.message = "Sentence generated successfully."
            self.get_logger().info(f"Generated: {sentence}")
        except Exception as e:
            response.success = False
            response.description = ""
            response.message = str(e)
            self.get_logger().error(response.message)

        return response

def main(args=None):
    rclpy.init(args=args)
    node = GuestSentenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
