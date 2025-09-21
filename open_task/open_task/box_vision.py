#!/usr/bin/env python3
import os
import base64
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
from robot_interfaces.srv import SpeakText
from openai import OpenAI
import cv2

class BoxVisionNode(Node):
    def __init__(self):
        super().__init__('box_vision_node')
        self.declare_parameter('camera_topic', '/camera/color/image_raw')
        self.declare_parameter('service_name', 'describe_box')
        # allow choosing which GPT model to use, defaulting to environment or gpt-4o
        default_model = os.getenv('OPENAI_VISION_MODEL', 'gpt-4o')
        self.declare_parameter('model', default_model)
        api_key = os.getenv('OPENAI_API_KEY', '')
        self.client = OpenAI(api_key=api_key)
        cam_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        srv_name = self.get_parameter('service_name').get_parameter_value().string_value
        self.model = self.get_parameter('model').get_parameter_value().string_value
        self.bridge = CvBridge()
        self.latest_img = None
        self.create_subscription(Image, cam_topic, self.image_callback, QoSPresetProfiles.SENSOR_DATA.value)
        self.create_service(Trigger, srv_name, self.handle_request)
        self.speak_cli = self.create_client(SpeakText, 'speak_text')
        self.get_logger().info('BoxVisionNode ready')

    def image_callback(self, msg: Image):
        self.latest_img = msg

    def handle_request(self, request: Trigger.Request, response: Trigger.Response):
        if self.latest_img is None:
            response.success = False
            response.message = 'no image available'
            return response
        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.latest_img, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'CV bridge failed: {e}')
            response.success = False
            response.message = 'conversion failed'
            return response
        ok, buf = cv2.imencode('.jpg', cv_img)
        if not ok:
            response.success = False
            response.message = 'encode failed'
            return response
        b64 = base64.b64encode(buf.tobytes()).decode('ascii')
        data_url = f'data:image/jpeg;base64,{b64}'
        messages = [
            {'role': 'user', 'content': [
                {'type': 'text', 'text': 'what can you see on the box'},
                {'type': 'image_url', 'image_url': {'url': data_url}}
            ]}
        ]
        try:
            resp = self.client.chat.completions.create(model=self.model, messages=messages)
            answer = resp.choices[0].message.content.strip()
        except Exception as e:
            self.get_logger().error(f'OpenAI call failed: {e}')
            response.success = False
            response.message = 'openai error'
            return response
        if self.speak_cli.wait_for_service(timeout_sec=1.0):
            req = SpeakText.Request()
            req.text = answer
            self.speak_cli.call_async(req)
        else:
            self.get_logger().warning('speak_text service not available')
        response.success = True
        response.message = answer
        return response

def main(args=None):
    rclpy.init(args=args)
    node = BoxVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()