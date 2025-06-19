#!/usr/bin/env python3
import os
import base64
import rclpy
from rclpy.node import Node
# Import QoS presets
from rclpy.qos import QoSPresetProfiles

# ROS message and service types
from std_msgs.msg import Int8, String
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatusArray
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from robot_interfaces.srv import SpeakText

# Utilities
from tf_transformations import quaternion_from_euler
from cv_bridge import CvBridge
import cv2

# YASMIN FSM
from yasmin import State, StateMachine, Blackboard
from yasmin_ros import set_ros_loggers
from yasmin_viewer import YasminViewerPub

# OpenAI client
from openai import OpenAI

class WaitAwakeState(State):
    def __init__(self):
        super().__init__(outcomes=['awake'])
        self.node = rclpy.create_node('wait_awake_state')
        self.awake = False
        # subscribe with default QoS
        self.node.create_subscription(
            Int8,
            '/awake_flag',
            self._cb,
            QoSPresetProfiles.SYSTEM_DEFAULT.value
        )

    def _cb(self, msg: Int8):
        if msg.data == 1:
            self.awake = True

    def execute(self, blackboard: Blackboard) -> str:
        while rclpy.ok() and not self.awake:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return 'awake'

class SpeakState(State):
    def __init__(self):
        super().__init__(outcomes=['succeeded', 'failed'])
        self.node = rclpy.create_node('speak_and_record_state_node')
        self.speak_cli = self.node.create_client(SpeakText, '/speak_text')
        self.record_cli = self.node.create_client(Trigger, '/record_audio')

    def execute(self, blackboard: Blackboard) -> str:
        if self.speak_cli.wait_for_service(timeout_sec=1.0):
            req = SpeakText.Request()
            req.text = "How can I help you?"
            fut = self.speak_cli.call_async(req)
            rclpy.spin_until_future_complete(self.node, fut)
            if fut.result() is None:
                self.node.get_logger().warning('speak_text call failed')
                return 'failed'
        else:
            self.node.get_logger().warning('speak_text service not available')
            return 'failed'
        if self.record_cli.wait_for_service(timeout_sec=1.0):
            trig_req = Trigger.Request()
            rec_fut = self.record_cli.call_async(trig_req)
            rclpy.spin_until_future_complete(self.node, rec_fut)
            if not (rec_fut.result() and rec_fut.result().success):
                self.node.get_logger().warning('record_audio service call failed')
        else:
            self.node.get_logger().warning('record_audio service not available')
        return 'succeeded'

class ListenCommandState(State):
    def __init__(self):
        super().__init__(outcomes=['received', 'failed'])
        self.node = rclpy.create_node('listen_command_state')
        self.received = None
        self.node.create_subscription(
            String,
            '/speech_recognition_transcript',
            self._cb,
            QoSPresetProfiles.SYSTEM_DEFAULT.value
        )
        self.speak_cli = self.node.create_client(SpeakText, '/speak_text')

    def _cb(self, msg: String):
        self.received = msg.data

    def execute(self, blackboard: Blackboard) -> str:
        while self.received is None and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)
        if self.received is None:
            return 'failed'
        blackboard['transcript'] = self.received
        if self.speak_cli.wait_for_service(timeout_sec=1.0):
            req = SpeakText.Request()
            req.text = "Okay, please wait while I check."
            fut = self.speak_cli.call_async(req)
            rclpy.spin_until_future_complete(self.node, fut)
        return 'received'

class NavigateState(State):
    def __init__(self):
        super().__init__(outcomes=['succeeded'])
        self.node = rclpy.create_node('navigate_state')
        self.pub = self.node.create_publisher(PoseStamped, '/goal_pose', 10)
        self.done = False
        self.node.create_subscription(
            GoalStatusArray,
            '/navigate_to_pose/_action/status',
            self._status_cb,
            QoSPresetProfiles.SYSTEM_DEFAULT.value
        )

    def _status_cb(self, msg: GoalStatusArray):
        if msg.status_list and msg.status_list[-1].status == 4:
            self.done = True

    def execute(self, blackboard: Blackboard) -> str:
        goal = PoseStamped()
        goal.header.stamp = self.node.get_clock().now().to_msg()
        goal.header.frame_id = 'map'
        goal.pose.position.x = 0.686044
        goal.pose.position.y = -0.89254
        goal.pose.position.z = 0.0
        qx, qy, qz, qw = quaternion_from_euler(0, 0, 0.0)
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw
        self.node.get_logger().info("Publishing navigation goal…")
        self.pub.publish(goal)
        while rclpy.ok() and not self.done:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        self.node.get_logger().info("Navigation succeeded.")
        return 'succeeded'

class VisualCommandState(State):
    def __init__(self):
        super().__init__(outcomes=['succeeded', 'failed'])
        self.node = rclpy.create_node('visual_and_speak_state')
        self.bridge = CvBridge()
        self.latest_img = None
        # QoS matching publisher reliability
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
        qos_sensor_data = QoSPresetProfiles.SENSOR_DATA.value
        self.node.create_subscription(
            Image,
            '/camera0/color/image_raw',
            self._image_cb,
            qos_sensor_data
        )
        # OpenAI and speaking clients
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"
        self.speak_cli = self.node.create_client(SpeakText, '/speak_text')

    def _image_cb(self, msg: Image):
        self.latest_img = msg

    def execute(self, blackboard: Blackboard) -> str:
        # wait up to 2 seconds for an image
        start = self.node.get_clock().now().nanoseconds
        timeout_ns = start + 2 * 1e9
        while rclpy.ok() and self.latest_img is None:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if self.node.get_clock().now().nanoseconds > timeout_ns:
                self.node.get_logger().error("No image received in time")
                return 'failed'
        # convert to OpenCV image (bgr8)
        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.latest_img, 'bgr8')
        except Exception as e:
            self.node.get_logger().error(f"CV bridge failed: {e}")
            return 'failed'
        # encode to JPEG + base64
        ok, jpeg = cv2.imencode('.jpg', cv_img)
        if not ok:
            self.node.get_logger().error("JPEG encoding failed")
            return 'failed'
        b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
        uri = f"data:image/jpeg;base64,{b64}"
        # prepare multimodal request
        user_cmd = blackboard['transcript']
        messages = [
            {"role": "system", "content": "You are a helpful vision assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": f"User command: {user_cmd}"},
                {"type": "image_url", "image_url": {"url": uri}}
            ]}
        ]
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
        except Exception as e:
            self.node.get_logger().error(f"OpenAI call failed: {e}")
            return 'failed'
        result = resp.choices[0].message.content
        blackboard['vision_result'] = result
        self.node.get_logger().info("Vision processing complete")
        # speak result
        if not self.speak_cli.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().warning('/speak_text service unavailable')
            return 'failed'
        req = SpeakText.Request()
        req.text = result
        fut = self.speak_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, fut)
        if fut.result() is None:
            self.node.get_logger().error("speak_text call failed")
            return 'failed'
        return 'succeeded'


def main():
    rclpy.init()
    set_ros_loggers()
    sm = StateMachine(outcomes=['succeeded', 'failed'])
    sm.add_state('WAIT_AWAKE', WaitAwakeState(), transitions={'awake': 'SPEAK'})
    sm.add_state('SPEAK', SpeakState(), transitions={'succeeded': 'LISTEN', 'failed': 'failed'})
    sm.add_state('LISTEN', ListenCommandState(), transitions={'received': 'NAVIGATE', 'failed': 'failed'})
    sm.add_state('NAVIGATE', NavigateState(), transitions={'succeeded': 'DETECT'})
    sm.add_state('DETECT', VisualCommandState(), transitions={'succeeded': 'succeeded', 'failed': 'failed'})
    YasminViewerPub('open_task_fsm', sm)
    try:
        result = sm()
        print(f"[MAIN] State machine finished with outcome: {result}")
    except KeyboardInterrupt:
        if sm.is_running(): sm.cancel_state()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
