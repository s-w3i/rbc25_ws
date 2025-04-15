#!/usr/bin/env python3
import os
import json
from openai import OpenAI
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import String

class HomeServiceRobot:
    def __init__(self, host_name="John"):
        """Initialize the home service robot.
        
        Args:
            host_name (str, optional): The host's name. Defaults to "John".
        """
        # Initialize the OpenAI client using the API key from the environment.
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.host_name = host_name
        # Conversation state variables.
        self.name = None
        self.drink = None
        self.conversation_active = False

        self.system_prompt = (
            "You are a home service robot assistant. Analyze the guest's input and:\n"
            " 1. Extract ALL mentioned entities: guest name and favourite drink.\n"
            " 2. Categorize the task as:\n"
            "    - \"Name\" (only guest name mentioned)\n"
            "    - \"Drink\" (only favourite drink mentioned)\n"
            "    - \"Both\" (both guest name and favourite drink mentioned)\n"
            "    - \"unknown\" (neither mentioned)\n\n"
            "Return JSON with:\n"
            "  - \"task\": the category,\n"
            "  - \"reason\": brief explanation,\n"
            "  - \"entities\": dictionary with:\n"
            "      - \"name\" (string or null)\n"
            "      - \"drink\" (string or null)\n\n"
            "Examples:\n"
            "Guest: \"I'm Alice and love coffee\" → \n"
            "{\n"
            "  \"task\": \"Both\",\n"
            "  \"reason\": \"Both guest name and favourite drink mentioned\",\n"
            "  \"entities\": {\"name\": \"Alice\", \"drink\": \"coffee\"}\n"
            "}\n"
            "Guest: \"Call me Bob\" → \n"
            "{\n"
            "  \"task\": \"Name\",\n"
            "  \"reason\": \"Only guest name mentioned\",\n"
            "  \"entities\": {\"name\": \"Bob\", \"drink\": null}\n"
            "}\n"
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
            temperature=0.9
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
            return f"Thank you, {name}! We have noted your favourite drink as {drink}. Please follow me."

    def start_conversation(self, send_response):
        """
        Start a new conversation by resetting conversation state and sending an initial greeting.
        
        Args:
            send_response (function): Callback to output a message.
        """
        self.name = None
        self.drink = None
        self.conversation_active = True
        send_response(f"Hello! Welcome to {self.host_name}'s house. May I please have your name and your favourite drink?")

    def process_input(self, user_text, send_response):
        """
        Process a user transcript message.
        
        Args:
            user_text (str): The received speech transcript.
            send_response (function): Callback to output a message.
        """
        # Only process if a conversation is active.
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
        send_response(f"{confirmation}")

        # End conversation if both details are collected.
        if self.name and self.drink:
            #send_response(f"Thank you, {self.name}! We have noted your favourite drink as {self.drink}. Please follow me.")
            self.conversation_active = False


class HomeServiceRobotNode(Node):
    def __init__(self):
        super().__init__('home_service_robot_node')
        self.get_logger().info("Home Service Robot Node started")
        self.robot = HomeServiceRobot(host_name="John")
        
        # Create a service to start the conversation.
        self.start_srv = self.create_service(Trigger, 'start_conversation', self.start_conversation_callback)
        self.get_logger().info("Service 'start_conversation' is ready.")

        # Create a publisher to send TTS/face commands to the /talk topic.
        self.talk_pub = self.create_publisher(String, 'talk', 10)

        # Create a subscriber to listen for speech recognition transcripts.
        self.create_subscription(String, 'speech_recognition_transcript', self.speech_callback, 10)
        self.get_logger().info("Subscriber to /speech_recognition_transcript is ready.")

    def send_response(self, message: str):
        """
        Log the message and publish it to the /talk topic.
        
        Args:
            message (str): The message to be published.
        """
        self.get_logger().info(message)
        msg = String()
        msg.data = message
        self.talk_pub.publish(msg)

    def start_conversation_callback(self, request, response):
        """
        Callback for the Trigger service to start the conversation.
        """
        self.robot.start_conversation(self.send_response)
        response.success = True
        response.message = "Conversation started. Awaiting speech transcripts on /speech_recognition_transcript."
        return response

    def speech_callback(self, msg: String):
        """
        Callback for processing incoming speech transcripts.
        """
        transcript = msg.data
        self.get_logger().info(f"Received speech transcript: {transcript}")
        self.robot.process_input(transcript, self.send_response)


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
