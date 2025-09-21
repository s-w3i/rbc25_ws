#!/usr/bin/env python3
import rclpy
import os
import time
import json
import cv2
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Vector3
from std_msgs.msg import Int8, String
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from tf_transformations import quaternion_from_euler
from robot_interfaces.srv import SpeakText, SegmentHumans, DescribeGuest, DescribeGuestSentence
from std_srvs.srv import Trigger, SetBool
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, DurabilityPolicy

from yasmin import State, Blackboard, StateMachine
from yasmin_ros import set_ros_loggers
from yasmin_viewer import YasminViewerPub

SEGMENT_DIR = "/home/usern/rbc25_ws/segmented_images"
if not os.path.exists(SEGMENT_DIR):
    os.makedirs(SEGMENT_DIR)


class GuestRegistry:
    """Simple container for guest information."""

    def __init__(self) -> None:
        self.guests = {}

    def add_guest(self, name: str, info: dict) -> None:
        self.guests[name] = info

    def get_guest(self, name: str):
        return self.guests.get(name)

class WaitAwakeState(State):
    """Block until /awake_flag publishes 1."""

    def __init__(self) -> None:
        super().__init__(["awake"])
        self.node = rclpy.create_node("wait_awake_state")
        self.awake = False
        self.node.create_subscription(Int8, "/awake_flag", self._cb, 10)

    def _cb(self, msg: Int8) -> None:
        if msg.data == 1:
            self.awake = True

    def execute(self, bb: Blackboard) -> str:
        self.node.get_logger().info("Waiting for /awake_flag == 1")
        while rclpy.ok() and not self.awake:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        speak_cli = self.node.create_client(SpeakText, '/speak_text')
        if speak_cli.wait_for_service(timeout_sec=2.0):
            req = SpeakText.Request()
            req.text = 'Hi, I will start my task now'
            fut = speak_cli.call_async(req)
            rclpy.spin_until_future_complete(self.node, fut)
            if fut.result() is None:
                self.node.get_logger().warning('speak_text call failed')
        else:
            self.node.get_logger().warning('/speak_text service not available')
        time.sleep(3.0)
        return "awake"

class StartConversationState(State):
    """Wait for /awake_flag then call /start_conversation and store guest info."""

    def __init__(self) -> None:
        super().__init__(["conversation_done"])
        self.guest_info = None
        self.guest_info_received = False

    def _guest_cb(self, msg: String) -> None:
        try:
            self.guest_info = json.loads(msg.data)
            self.guest_info_received = True
        except Exception as e:
            self.node.get_logger().error(f"Failed to parse guest info: {e}")

    def execute(self, bb: Blackboard) -> str:
        self.guest_info = None
        self.guest_info_received = False
        self.node = rclpy.create_node("start_conversation")

        client = self.node.create_client(Trigger, "/start_conversation")
        if client.wait_for_service(timeout_sec=5.0):
            fut = client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self.node, fut)
        else:
            self.node.get_logger().error("/start_conversation not available")

        self.node.create_subscription(String, "/guest_info", self._guest_cb, 10)
        self.node.get_logger().info("Waiting for guest info...")
        while rclpy.ok() and not self.guest_info_received:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        if self.guest_info:
            # Retrieve or create the guest registry on the blackboard
            if "guest_registry" in bb:
                reg = bb["guest_registry"]
            else:
                reg = GuestRegistry()
                bb["guest_registry"] = reg

            name = self.guest_info.get("name", "unknown")
            reg.add_guest(name, self.guest_info)
            bb["current_guest"] = self.guest_info

        self.node.destroy_node()
        return "conversation_done"


class RegisterGuestState(State):
    """Capture and describe the current guest."""

    def __init__(self) -> None:
        super().__init__(["registered"])
        self.bridge = CvBridge()

    def execute(self, bb: Blackboard) -> str:
        self.node = rclpy.create_node("register_guest")
        spk = self.node.create_client(SpeakText, "/speak_text")
        if spk.wait_for_service(timeout_sec=5.0):
            req = SpeakText.Request()
            req.text = "Taking your picture to register you into the system."
            fut = spk.call_async(req)
            rclpy.spin_until_future_complete(self.node, fut)

        seg_cli = self.node.create_client(SegmentHumans, "/segment_humans")
        if not seg_cli.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().error("/segment_humans not available")
            self.node.destroy_node()
            return "registered"

        res = None
        for attempt in range(3):
            fut_seg = seg_cli.call_async(SegmentHumans.Request())
            rclpy.spin_until_future_complete(self.node, fut_seg)
            res = fut_seg.result()
            if res and res.success and res.segmented_images:
                break
            self.node.get_logger().warning(
                f"Segmentation failed on attempt {attempt + 1}, retrying..."
            )
            time.sleep(1)

        if not res or not res.success or not res.segmented_images:
            self.node.get_logger().error("Segmentation failed after retries")
            self.node.destroy_node()
            return "registered"

        img = res.segmented_images[0]
        try:
            cv_img = self.bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")
        except Exception as e:
            self.node.get_logger().error(f"Image conversion error: {e}")
            self.node.destroy_node()
            return "registered"

        if "current_guest" in bb:
            info = bb["current_guest"]
        else:
            info = {}
        name = info.get("name", "unknown")
        path = os.path.join(SEGMENT_DIR, f"{name}.jpg")
        cv2.imwrite(path, cv_img)
        info["segmented_file"] = path

        desc_cli = self.node.create_client(DescribeGuest, "/describe_guest")
        if desc_cli.wait_for_service(timeout_sec=5.0):
            dreq = DescribeGuest.Request()
            dreq.image_path = path
            fut_d = desc_cli.call_async(dreq)
            rclpy.spin_until_future_complete(self.node, fut_d)
            dres = fut_d.result()
            if dres and dres.success:
                try:
                    parsed = json.loads(dres.description)
                    info.update(parsed)
                except Exception as e:
                    self.node.get_logger().error(f"Describe parse error: {e}")

        bb["guest_registry"].add_guest(name, info)

        spk2 = self.node.create_client(SpeakText, "/speak_text")
        if spk2.wait_for_service(timeout_sec=5.0):
            req2 = SpeakText.Request()
            req2.text = (
                f"hi {name}, i had register you into the system, please follow me"
            )
            fut2 = spk2.call_async(req2)
            rclpy.spin_until_future_complete(self.node, fut2)

        self.node.destroy_node()
        return "registered"

class GoToPoseState(State):
    """Navigate the robot to a predefined pose."""

    def __init__(self, pose, name="go_to_pose") -> None:
        super().__init__(["done"])
        self.pose = pose
        self.node = None
        self.name = name

    def execute(self, bb: Blackboard) -> str:
        node = rclpy.create_node(self.name)
        self.node = node

        ac = ActionClient(node, NavigateToPose, 'navigate_to_pose')
        if not ac.wait_for_server(timeout_sec=10.0):
            node.get_logger().error('NavigateToPose action server not available')
            node.destroy_node()
            return 'done'

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.stamp = node.get_clock().now().to_msg()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose.position = Point(x=self.pose['x'], y=self.pose['y'], z=0.0)
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, self.pose['yaw'])
        goal_msg.pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)

        node.get_logger().info(f'Sending goal: {self.pose}')
        send_goal_future = ac.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(node, send_goal_future)
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            node.get_logger().error('Goal was rejected')
            node.destroy_node()
            return 'done'

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(node, result_future)
        result = result_future.result()
        if result.status == GoalStatus.STATUS_SUCCEEDED:
            node.get_logger().info('Navigation succeeded')
        else:
            node.get_logger().error(f'Navigation failed with status {result.status}')
        node.destroy_node()
        return "done"

class SpeakWelcomeState(State):
    """Announce arrival at the first guest location."""

    def __init__(self) -> None:
        super().__init__(["done"])

    def execute(self, bb: Blackboard) -> str:
        node = rclpy.create_node("speak_welcome")
        cli = node.create_client(SpeakText, "/speak_text")
        if cli.wait_for_service(timeout_sec=2.0):
            req = SpeakText.Request()
            req.text = "welcome to max house"
            fut = cli.call_async(req)
            rclpy.spin_until_future_complete(node, fut)
            if fut.result() is None:
                node.get_logger().warning("speak_text call failed")
        else:
            node.get_logger().warning("/speak_text service not available")
        node.destroy_node()
        return "done"
    
class IntroduceGuestToHostState(State):
    """Introduce the registered guest to the host using a description."""

    def __init__(self) -> None:
        super().__init__(["introduced"])

    def execute(self, bb: Blackboard) -> str:
        node = rclpy.create_node("introduce_guest_to_host")

        guest_info = bb["current_guest"]
        if not guest_info:
            node.get_logger().error("No guest info available for introduction")
            node.destroy_node()
            return "introduced"

        host_name = "Max"
        guest_name = guest_info.get("name", "unknown")

        raw_sentence = None
        desc_cli = node.create_client(DescribeGuestSentence, "/describe_guest_sentence")
        if desc_cli.wait_for_service(timeout_sec=5.0):
            req = DescribeGuestSentence.Request()
            req.guest_info_json = json.dumps(guest_info, ensure_ascii=False)
            fut = desc_cli.call_async(req)
            rclpy.spin_until_future_complete(node, fut)
            res = fut.result()
            if res and res.success:
                raw_sentence = res.description
        else:
            node.get_logger().error("/describe_guest_sentence service not available")

        if raw_sentence:
            raw_sentence = raw_sentence.rstrip('.')
            text = f"Hi {host_name}, this is {guest_name}, {raw_sentence}."
        else:
            text = f"Hi {host_name}, this is {guest_name}."

        spk = node.create_client(SpeakText, "/speak_text")
        if spk.wait_for_service(timeout_sec=5.0):
            req = SpeakText.Request()
            req.text = text
            fut = spk.call_async(req)
            rclpy.spin_until_future_complete(node, fut)
        else:
            node.get_logger().error("/speak_text service not available to introduce guest")

        node.destroy_node()
        return "introduced"

class EmptyChairDetectionState(State):
    """Announce and find an empty chair for the current guest."""

    def __init__(self) -> None:
        super().__init__(["done"])

    def execute(self, bb: Blackboard) -> str:
        node = rclpy.create_node("empty_chair_detection")

        if "current_guest" not in bb:
            node.get_logger().error("No current guest info available!")
            node.destroy_node()
            return "done"

        guest = bb["current_guest"]
        guest_name = guest.get("name", "unknown")

        speak_client = node.create_client(SpeakText, "/speak_text")
        if speak_client.wait_for_service(timeout_sec=5.0):
            req = SpeakText.Request()
            req.text = f"Hi {guest_name}, I will find you an empty seat"
            fut = speak_client.call_async(req)
            rclpy.spin_until_future_complete(node, fut)
        else:
            node.get_logger().error("/speak_text not available for seat search")

        detect_client = node.create_client(Trigger, "detect_empty_chair")
        if not detect_client.wait_for_service(timeout_sec=5.0):
            node.get_logger().error("detect_empty_chair service not available")
        else:
            success = False
            for attempt in range(3):
                req = Trigger.Request()
                fut = detect_client.call_async(req)
                rclpy.spin_until_future_complete(node, fut)
                res = fut.result()
                if res and res.success:
                    success = True
                    node.get_logger().info(
                        f"Empty chair detected: {res.message}"
                    )
                    assign_cli = node.create_client(SpeakText, "/speak_text")
                    if assign_cli.wait_for_service(timeout_sec=5.0):
                        areq = SpeakText.Request()
                        areq.text = (
                            f"Hi {guest_name}, you may have your seat here"
                        )
                        afut = assign_cli.call_async(areq)
                        rclpy.spin_until_future_complete(node, afut)
                    break
                else:
                    node.get_logger().warning(
                        f"Empty chair detection attempt {attempt + 1} failed"
                    )
                    time.sleep(1)
            if not success:
                node.get_logger().error(
                    "Failed to detect empty chair after retries"
                )
                
        qos_profile = QoSProfile(depth=10)
        qos_profile.durability = DurabilityPolicy.TRANSIENT_LOCAL  

        arm_pub = node.create_publisher(Vector3, "/arm_target", qos_profile)
        arm_msg = Vector3()
        arm_msg.x = 0.05
        arm_msg.y = 0.0
        arm_msg.z = 0.1
        arm_pub.publish(arm_msg)
        node.get_logger().info("Published arm target at /arm_target: [0.05, 0.0, 0.1]")

        time.sleep(5)
        node.destroy_node()
        return "done"

class WaitAwakeAgainState(State):
    """Wait for another /awake_flag == 1 without speaking."""

    def __init__(self) -> None:
        super().__init__(["awake"])
        self.node = rclpy.create_node("wait_awake_again_state")
        self.awake = False
        self.node.create_subscription(Int8, "/awake_flag", self._cb, 10)

    def _cb(self, msg: Int8) -> None:
        if msg.data == 1:
            self.awake = True

    def execute(self, bb: Blackboard) -> str:
        self.node.get_logger().info("Waiting again for /awake_flag == 1")
        while rclpy.ok() and not self.awake:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return "awake"

class AskHelpState(State):
    """After the second awake flag, greet and record a request."""

    def __init__(self) -> None:
        super().__init__(["done"])
        self.transcript = None

    def _transcript_cb(self, msg: String) -> None:
        self.transcript = msg.data

    def execute(self, bb: Blackboard) -> str:
        node = rclpy.create_node("ask_help_state")

        speak_cli = node.create_client(SpeakText, "/speak_text")
        if speak_cli.wait_for_service(timeout_sec=5.0):
            req = SpeakText.Request()
            req.text = "Hi, What can I help you"
            fut = speak_cli.call_async(req)
            rclpy.spin_until_future_complete(node, fut)
        else:
            node.get_logger().error("/speak_text service not available")

        record_cli = node.create_client(Trigger, "record_audio")
        if record_cli.wait_for_service(timeout_sec=5.0):
            rclpy.spin_until_future_complete(node, record_cli.call_async(Trigger.Request()))
        else:
            node.get_logger().error("record_audio service not available")

        node.create_subscription(String, "speech_recognition_transcript", self._transcript_cb, 10)
        node.get_logger().info("Waiting for speech recognition transcript...")
        while rclpy.ok() and self.transcript is None:
            rclpy.spin_once(node, timeout_sec=0.1)

        if self.transcript:
            node.get_logger().info(f"Transcript received: {self.transcript}")
        else:
            node.get_logger().warning("No transcript received")

        if speak_cli.service_is_ready():
            req2 = SpeakText.Request()
            req2.text = "ok, I will carry out your request"
            fut2 = speak_cli.call_async(req2)
            rclpy.spin_until_future_complete(node, fut2)
        else:
            node.get_logger().warning("/speak_text service not ready for response")

        node.destroy_node()
        return "done"
    
class DescribeBoxState(State):
    """Trigger the describe_box service to summarize the box contents."""

    def __init__(self) -> None:
        super().__init__(["done"])

    def execute(self, bb: Blackboard) -> str:
        node = rclpy.create_node("describe_box_state")
        
        # arm_pub = node.create_publisher(Vector3, "/arm_target", 10)
        # arm_msg = Vector3(x=0.05, y=0.0, z=0.05)
        # arm_pub.publish(arm_msg)
        # node.get_logger().info(f"Published arm target {arm_msg}")
        
        # time.sleep(3)  # Wait for the arm to move

        cli = node.create_client(Trigger, "describe_box")
        if cli.wait_for_service(timeout_sec=5.0):
            fut = cli.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(node, fut)
            res = fut.result()
            if not res or not res.success:
                node.get_logger().error("describe_box service failed")
        else:
            node.get_logger().error("describe_box service not available")

        node.destroy_node()
        return "done"
    
class PlaceBackState(State):
    """Return the arm and open the gripper after the second help."""

    def __init__(self) -> None:
        super().__init__(["done"])
        
    def execute(self, bb: Blackboard) -> str:
        node = rclpy.create_node(f"place_back_state_{int(time.time()*1000)}")
        qos_profile = QoSProfile(depth=10)
        qos_profile.durability = DurabilityPolicy.TRANSIENT_LOCAL  

        arm_pub = node.create_publisher(Vector3, "/arm_target", qos_profile)

        def pub(x, y, z):
            msg = Vector3(x=x, y=y, z=z)
            arm_pub.publish(msg)
            node.get_logger().info(f"Published arm target {msg}")

        pub(0.35, -0.10, 0.10)
        time.sleep(5)

        grip_cli = node.create_client(SetBool, "gripper_control")
        if grip_cli.wait_for_service(timeout_sec=5.0):
            req = SetBool.Request()
            req.data = True
            fut = grip_cli.call_async(req)
            rclpy.spin_until_future_complete(node, fut)
        else:
            node.get_logger().error("gripper_control service not available")

        time.sleep(5)  # Wait for the gripper to close
        pub(0.35, -0.10, 0.20)
        time.sleep(5)
        pub(0.2, -0.35, 0.2)
        time.sleep(5)
        if grip_cli.wait_for_service(timeout_sec=5.0):
            req = SetBool.Request()
            req.data = False
            fut = grip_cli.call_async(req)
            rclpy.spin_until_future_complete(node, fut)
        else:
            node.get_logger().error("gripper_control service not available")
        
        pub(0.05, 0.0, 0.05)
        time.sleep(5)

        node.destroy_node()
        return "done"

class Pass(State):

    def __init__(self) -> None:
        super().__init__(["done"])
        
    def execute(self, bb: Blackboard) -> str:
        node = rclpy.create_node(f"place_back_state_{int(time.time()*1000)}")
        qos_profile = QoSProfile(depth=10)
        qos_profile.durability = DurabilityPolicy.TRANSIENT_LOCAL  

        arm_pub = node.create_publisher(Vector3, "/arm_target", qos_profile)

        def pub(x, y, z):
            msg = Vector3(x=x, y=y, z=z)
            arm_pub.publish(msg)
            node.get_logger().info(f"Published arm target {msg}")

        pub(0.35, 0.0, 0.20)
        time.sleep(5)

        grip_cli = node.create_client(SetBool, "gripper_control")
        if grip_cli.wait_for_service(timeout_sec=5.0):
            req = SetBool.Request()
            req.data = False
            fut = grip_cli.call_async(req)
            rclpy.spin_until_future_complete(node, fut)
        else:
            node.get_logger().error("gripper_control service not available")

        time.sleep(5)  # Wait for the gripper to open
        pub(0.05, 0.0, 0.05)
        time.sleep(5)

        node.destroy_node()
        return "done"

def main():
    rclpy.init()
    set_ros_loggers()

    # Example poses from task_state2.py
    guest_pose = {'x': 0.25, 'y': -4.1, 'yaw': 3.142}
    host_pose = {'x': -0.4, 'y': -1.2, 'yaw': 0.0}
    pick_pose = {'x': 0.60, 'y': 0.0, 'yaw': 0.0}
    host_pose2 = {'x': -0.3, 'y': 0.0, 'yaw': 0.0}

    sm = StateMachine(outcomes=["finished"])
    sm.add_state("WAIT", WaitAwakeState(), transitions={"awake": "AWAKE2"})
    sm.add_state("GUEST1", GoToPoseState(guest_pose, "go_to_guest1"),
                 transitions={"done": "WELCOME"})
    sm.add_state("WELCOME", SpeakWelcomeState(), transitions={"done": "CONVERSATION"})
    sm.add_state("CONVERSATION", StartConversationState(),
                 transitions={"conversation_done": "REGISTER"})
    sm.add_state("REGISTER", RegisterGuestState(), transitions={"registered": "HOST"})
    sm.add_state("HOST", GoToPoseState(host_pose, "go_to_host"),
                 transitions={"done": "INTRODUCE_TO_HOST"})
    sm.add_state("INTRODUCE_TO_HOST", IntroduceGuestToHostState(),
                 transitions={"introduced": "EMPTY_CHAIR"}),
    sm.add_state("EMPTY_CHAIR", EmptyChairDetectionState(),
                 transitions={"done": "finished"})
    sm.add_state("AWAKE2", WaitAwakeAgainState(), transitions={"awake": "ASK_HELP"})
    sm.add_state("ASK_HELP", AskHelpState(), transitions={"done": "DESCRIBE_BOX"})
    sm.add_state("PICK", GoToPoseState(pick_pose, "go_to_pick"),
                 transitions={"done": "DESCRIBE_BOX"})
    sm.add_state("DESCRIBE_BOX", DescribeBoxState(), transitions={"done": "AWAKE3"})
    sm.add_state("AWAKE3", WaitAwakeAgainState(), transitions={"awake": "ASK_HELP2"})
    sm.add_state("ASK_HELP2", AskHelpState(), transitions={"done": "PLACE_BACK"})
    sm.add_state("PLACE_BACK", PlaceBackState(), transitions={"done": "finished"})
    # sm.add_state("GIVE", GoToPoseState(host_pose2, "go_to_host"),
    #              transitions={"done": "PASS"})
    #sm.add_state("PASS", Pass(), transitions={"done": "finished"})
    
    YasminViewerPub("three_point_nav", sm)

    bb = Blackboard()
    try:
        outcome = sm(bb)
        print("State machine finished with outcome:", outcome)
    except KeyboardInterrupt:
        if sm.is_running():
            sm.cancel_state()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()