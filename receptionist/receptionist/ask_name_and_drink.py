#!/usr/bin/env python3
import os
import json
import time

from openai import OpenAI
import rclpy
from rclpy.node import Node

# Standard trigger for "start_conversation" and "record_audio" 
from std_srvs.srv import Trigger
from std_msgs.msg import String

# Custom service for TTS
from robot_interfaces.srv import SpeakText


class HomeServiceRobot:
    def __init__(self, host_name="Max"):
        """Initialize the home service robot."""
        # Initialize the OpenAI client.
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.host_name = host_name
        # Conversation state variables.
        self.name = None
        self.drink = None
        self.conversation_active = False

        self.system_prompt = (
        "You are a home service robot assistant. Analyze the guest's input and:\n"
        "1. Extract the guest's name as stated (any name is acceptable).\n"
        "2. For the drink:\n"
        "   a. First check if the mentioned drink is exactly one of the allowed options below.\n"
        "   b. If not, guess the most probable drink from this list based on similarity or intent.\n"
        "\n"
        "Allowed Drinks: coke, green tea, wine, orange juice, coffee, soda, cocoa, lemonade, coconut milk, black  tea.\n"
        "\n"
        "3. Categorize the task as:\n"
        "   - \"Name\"\n"
        "   - \"Drink\"\n"
        "   - \"Both\"\n"
        "   - \"unknown\"\n"
        "\n"
        "Return a JSON object with:\n"
        "  - \"task\": category\n"
        "  - \"reason\": short explanation\n"
        "  - \"entities\": {\n"
        "      \"name\": extracted name or null,\n"
        "      \"drink\": matched or guessed drink from allowed list, or null\n"
        "    }\n"
    )

    def classify(self, text):
        """Classify the guest's input and return a structured JSON response."""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)

    def generate_confirmation_sentence(self, name=None, drink=None):
        """Generate a concise confirmation sentence based on the gathered details."""
        if not name and not drink:
            return "Could you please provide both your name and your favourite drink?"
        elif name and not drink:
            return f"Hello {name}, may I have your favourite drink, please?"
        elif drink and not name:
            return f"You mentioned {drink} as your favourite drink. Could you please tell me your name?"
        else:
            return f"Thank you, {name}! We have noted your favourite drink as {drink}."

    def start_conversation(self, send_response):
        """
        Start a new conversation by resetting conversation state and sending an initial greeting.
        """
        self.name = None
        self.drink = None
        self.conversation_active = True
        send_response(
            f"Hello! Welcome to {self.host_name}'s house. May I please have your name and your favourite drink?"
        )

    def process_input(self, user_text, send_response):
        """
        Process a user transcript message.
        """
        if not self.conversation_active:
            return

        result = self.classify(user_text)
        new_name = result.get("entities", {}).get("name")
        new_drink = result.get("entities", {}).get("drink")

        if new_name and new_name.lower() not in ["null", "none", ""]:
            self.name = new_name
        if new_drink and new_drink.lower() not in ["null", "none", ""]:
            self.drink = new_drink

        confirmation = self.generate_confirmation_sentence(self.name, self.drink)
        send_response(confirmation)

        # End conversation if both details are collected.
        if self.name and self.drink:
            self.conversation_active = False


class HomeServiceRobotNode(Node):
    def __init__(self):
        super().__init__('home_service_robot_node')
        self.get_logger().info("Home Service Robot Node started")
        self.robot = HomeServiceRobot(host_name="Max")
        
        # 1) Service to start conversation.
        self.start_srv = self.create_service(Trigger, 'start_conversation', self.start_conversation_callback)
        self.get_logger().info("Service 'start_conversation' is ready.")

        # 2) Subscription to speech transcripts.
        self.create_subscription(String, 'speech_recognition_transcript', self.speech_callback, 10)
        self.get_logger().info("Subscriber to /speech_recognition_transcript is ready.")

        # 3) Client for record_audio.
        self.record_cli = self.create_client(Trigger, 'record_audio')
        self.get_logger().info("Client for 'record_audio' created.")

        # 4) Client for speak_text (custom service).
        self.speak_cli = self.create_client(SpeakText, 'speak_text')
        self.get_logger().info("Client for 'speak_text' created.")

        # 5) Publisher for exporting guest information.
        self.guest_pub = self.create_publisher(String, 'guest_info', 10)
        self.get_logger().info("Publisher for 'guest_info' created.")

        # 6) Internal flag to avoid republishing guest info more than once.
        self.guest_info_published = False

    def send_response(self, message: str):
        """
        Call speak_text with 'message'. Once speak_text returns success,
        record_audio will be triggered by the speak_text response callback.
        """
        self.get_logger().info(f"[send_response] {message}")
        if self.speak_cli.service_is_ready():
            req = SpeakText.Request()
            req.text = message
            future = self.speak_cli.call_async(req)
            future.add_done_callback(self._speak_response_cb)
        else:
            self.get_logger().warn("speak_text service not available yet.")

    def _speak_response_cb(self, future):
        """
        Callback after the speak_text service is called.
        If speak_text returns success, then trigger record_audio.
        """
        try:
            result = future.result()
            if result.success:
                self.get_logger().info(f"Succeeded speaking text: {result.message}")
                if self.record_cli.service_is_ready():
                    rec_req = Trigger.Request()
                    rec_future = self.record_cli.call_async(rec_req)
                    rec_future.add_done_callback(self._record_audio_response_cb)
                else:
                    self.get_logger().warn("record_audio service not available yet.")
            else:
                self.get_logger().warn(f"Failed to speak text: {result.message}")
        except Exception as e:
            self.get_logger().error(f"Exception calling speak_text service: {str(e)}")

    def _record_audio_response_cb(self, future):
        """
        Handle the response from the record_audio service.
        """
        try:
            result = future.result()
            if result.success:
                self.get_logger().info(f"Recording triggered: {result.message}")
            else:
                self.get_logger().warn(f"Failed to start recording: {result.message}")
        except Exception as e:
            self.get_logger().error(f"Exception calling record_audio service: {str(e)}")

    def start_conversation_callback(self, request, response):
        """
        Callback for the start_conversation service.
        """
        self.robot.start_conversation(self.send_response)
        response.success = True
        response.message = "Conversation started. Awaiting speech transcripts on /speech_recognition_transcript."
        self.guest_info_published = False
        return response

    def speech_callback(self, msg: String):
        """
        Callback for processing incoming speech transcripts.
        """
        transcript = msg.data
        self.get_logger().info(f"Received speech transcript: {transcript}")
        self.robot.process_input(transcript, self.send_response)

        # If both name and drink are collected and not yet exported, publish them.
        if not self.robot.conversation_active and not self.guest_info_published:
            if self.robot.name and self.robot.drink:
                guest_info = {"name": self.robot.name, "drink": self.robot.drink}
                info_msg = String()
                info_msg.data = json.dumps(guest_info)
                self.guest_pub.publish(info_msg)
                self.get_logger().info(f"Exported guest information: {info_msg.data}")
                self.guest_info_published = True

def main(args=None):
    rclpy.init(args=args)
    node = HomeServiceRobotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Home Service Robot Node")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
