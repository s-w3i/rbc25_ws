
#!/usr/bin/env python3
import os
import base64
import threading
import cv2
import requests
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from std_msgs.msg import String, Int8
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# import the SpeakText service definition
from robot_interfaces.srv import SpeakText

class ClassifierNode(Node):
    def __init__(self):
        super().__init__('classifier_node')

        # --- parameters ---
        self.declare_parameter('openai_api_key', os.getenv('OPENAI_API_KEY', ''))
        api_key = self.get_parameter('openai_api_key').get_parameter_value().string_value

        # OpenAI HTTP client
        self.client = OpenAI(api_key=api_key)

        # LangChain LLM and memory
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            openai_api_key=api_key,
            max_tokens=150
        )
        self.memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True
        )
        self.prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=(
                "You are a concise home assistant. Answer in one or two sentences with no extra commentary.\n"
                "Conversation so far:\n{history}\n"
                "User: {input}\n"
                "Assistant:"
            )
        )
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=self.prompt,
            verbose=False
        )

        # CV Bridge
        self.bridge = CvBridge()
        self.frame_lock = threading.Lock()
        self.latest_frame = None

        # SpeakText and record_audio service clients
        self.speak_client = self.create_client(SpeakText, 'speak_text')
        self.record_cli = self.create_client(Trigger, 'record_audio')

        # ROS topics
        self.declare_parameter('camera_topic', '/camera0/color/image_raw')
        self.declare_parameter('command_topic', '/speech_recognition_transcript')
        self.declare_parameter('response_topic', '/analysis_result')

        cam_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        cmd_topic = self.get_parameter('command_topic').get_parameter_value().string_value
        res_topic = self.get_parameter('response_topic').get_parameter_value().string_value

        # Subscriptions and publications
        self.pub = self.create_publisher(String, res_topic, 10)
        self.create_subscription(String, cmd_topic, self.command_callback, 10)
        self.create_subscription(
            Image,
            cam_topic,
            self.image_callback,
            QoSPresetProfiles.SENSOR_DATA.value
        )
        self.create_subscription(
            Int8,
            '/awake_flag',
            self.awake_callback,
            10
        )

        self.get_logger().info('ClassifierNode with DALL·E, vision, speech, and wake support ready.')

    def image_callback(self, msg: Image):
        # Update latest camera frame
        with self.frame_lock:
            self.latest_frame = msg

    def awake_callback(self, msg: Int8):
        self.get_logger().info(f'Awake flag received: {msg.data}')
        # send speak_text and only after it's done, trigger record_audio
        if self.speak_client.wait_for_service(timeout_sec=1.0):
            speak_req = SpeakText.Request()
            speak_req.text = "How can I help you?"
            future = self.speak_client.call_async(speak_req)
            future.add_done_callback(self._after_speak_record)
        else:
            self.get_logger().warning('speak_text service not available')

    def _after_speak_record(self, future):
        # once speech is done, trigger recording
        if self.record_cli.wait_for_service(timeout_sec=1.0):
            trigger_req = Trigger.Request()
            self.record_cli.call_async(trigger_req)
            self.get_logger().info('record_audio service triggered')
        else:
            self.get_logger().warning('record_audio service not available')

    def command_callback(self, msg: String):
        text = msg.data.strip()
        self.get_logger().info(f"User ► {text}")

        # Route request
        if self.needs_dalle(text):
            reply = self.handle_dalle_request(text)
        elif self.needs_vision(text):
            reply = self.handle_vision_request(text)
            self.memory.save_context({"input": text}, {"output": reply})
            self._speak(reply)
        else:
            reply = self.conversation.predict(input=text)
            self.memory.save_context({"input": text}, {"output": reply})
            self._speak(reply)

        # Publish to /analysis_result
        out = String(data=reply)
        self.pub.publish(out)
        self.get_logger().info(f"Bot ► {reply}")

    def _speak(self, text: str):
        # Helper to call speak_text service
        if self.speak_client.wait_for_service(timeout_sec=1.0):
            req = SpeakText.Request()
            req.text = text
            self.speak_client.call_async(req)
        else:
            self.get_logger().warning('speak_text service not available')

    def needs_vision(self, text: str) -> bool:
        low = text.lower()
        return any(kw in low for kw in ('see', 'look', 'detect', 'how many', 'count', 'what is this'))

    def needs_dalle(self, text: str) -> bool:
        low = text.lower()
        return any(kw in low for kw in ('draw', 'sketch', 'generate image', 'create picture', 'picture of'))

    def handle_vision_request(self, text: str) -> str:
        with self.frame_lock:
            img_msg = self.latest_frame
        if not img_msg:
            return "I can't see anything right now."
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        h, w = cv_img.shape[:2]
        if max(h, w) > 256:
            scale = 256 / max(h, w)
            cv_img = cv2.resize(cv_img, (int(w*scale), int(h*scale)))
        ok, buf = cv2.imencode('.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 20])
        if not ok:
            return "Sorry, I couldn't process the image."
        b64 = base64.b64encode(buf.tobytes()).decode('ascii')
        data_url = f"data:image/jpeg;base64,{b64}"
        system_msg = {'role': 'system', 'content': 'Use the image and question to answer succinctly.'}
        user_msg = {'role': 'user', 'content': [
            {'type': 'text', 'text': text},
            {'type': 'image_url', 'image_url': {'url': data_url}}
        ]}
        resp = self.client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[system_msg, user_msg]
        )
        return resp.choices[0].message.content.strip()

    def handle_dalle_request(self, text: str) -> str:
        low = text.lower()
        subject = text.strip()
        for phrase in ('draw me a', 'draw a', 'sketch a', 'generate image of', 'create picture of', 'picture of'):
            if phrase in low:
                subject = low.split(phrase, 1)[1].strip()
                break
        subject = subject.rstrip('?.!')
        try:
            gen = self.client.images.generate(
                model="dall-e-3",
                prompt=text,
                n=1,
                size="1024x1024",
                response_format="url"
            )
            url = gen.data[0].url
            resp_img = requests.get(url)
            arr = np.frombuffer(resp_img.content, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                reply = "Failed to decode generated image."
            else:
                reply = f"Here is the generated image for {subject}."
                cv2.imshow("Generated Image", img)
                self._speak(reply)
                cv2.waitKey(10000)
                cv2.destroyWindow("Generated Image")
            self.memory.save_context({"input": text}, {"output": reply})
            return reply
        except Exception as e:
            self.get_logger().error(f"DALL·E API error: {e}")
            return "Sorry, I couldn't generate the image right now."


def main(args=None):
    rclpy.init(args=args)
    node = ClassifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()