#!/usr/bin/env python3
"""
follow_me_state_machine.py

• Waits for /awake_flag (Int8) to start a voice interaction
• Starts people-following when the user says “follow me”
• Ends following when the user later says “stop follow me”
• Robust against repeated /awake_flag triggers
"""

import rclpy, time
from yasmin           import State, Blackboard, StateMachine
from yasmin_ros       import set_ros_loggers
from yasmin_viewer    import YasminViewerPub

from std_msgs.msg     import Int8, String
from std_srvs.srv     import Trigger, SetBool
from robot_interfaces.srv import SpeakText           # ← your TTS service
from geometry_msgs.msg import Vector3
from action_msgs.srv import CancelGoal
from action_msgs.msg import GoalInfo
import tf2_ros
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
import subprocess
from geometry_msgs.msg import Point

# ------------  Small helpers --------------------------------------------------
def _wait_for_service(node, client, name):
    if not client.wait_for_service(timeout_sec=3.0):
        node.get_logger().error(f"Service {name} not available.")
        return False
    return True


def _call_trigger(node, srv_name):
    cli = node.create_client(Trigger, srv_name)
    if _wait_for_service(node, cli, srv_name):
        fut = cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, fut)
        return fut.result() and fut.result().success
    return False


def _call_set_bool(node, srv_name, value):
    cli = node.create_client(SetBool, srv_name)
    if _wait_for_service(node, cli, srv_name):
        req       = SetBool.Request()
        req.data  = value
        fut       = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut)
        return fut.result() and fut.result().success
    return False


# ------------------------------------------------------------------------------
class WaitAwakePrompt(State):
    """Step 1 – trigger on /awake_flag, store pose, greet, start recording, enable detection"""
    def __init__(self):
        super().__init__(["awake"])

    def execute(self, bb):
        node             = rclpy.create_node("wait_awake_prompt")
        self.awake_recvd = False

        # Listen for awake flag
        node.create_subscription(
            Int8, "/awake_flag",
            lambda m: setattr(self, "awake_recvd", True),
            10
        )

        node.get_logger().info("Waiting for /awake_flag …")
        while rclpy.ok() and not self.awake_recvd:
            rclpy.spin_once(node, timeout_sec=0.2)

        # --- Initialize TF listener and wait for transform ---
        tf_buffer   = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer, node)
        start_time  = node.get_clock().now()
        timeout     = rclpy.duration.Duration(seconds=2.0)
        node.get_logger().info("Waiting for TF transform map->base_footprint…")
        while rclpy.ok():
            if tf_buffer.can_transform('map', 'base_footprint', rclpy.time.Time()):
                break
            if node.get_clock().now() - start_time > timeout:
                node.get_logger().error("Timeout waiting for TF map->base_footprint")
                break
            rclpy.spin_once(node, timeout_sec=0.1)

        try:
            transform = tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time()
            )
            bb.start_transform = transform
            tx = transform.transform.translation
            node.get_logger().info(
                f"Stored start pose: [{tx.x:.3f}, {tx.y:.3f}, {tx.z:.3f}]"
            )
        except Exception as e:
            node.get_logger().error(f"TF lookup failed after wait: {e}")
            
        arm_pub = node.create_publisher(Vector3, '/arm_target', 10)
        home = Vector3(x=0.05, y=0.0, z=0.1)
        arm_pub.publish(home)

        # 1) greet
        spk = node.create_client(SpeakText, "/speak_text")
        if _wait_for_service(node, spk, "/speak_text"):
            req = SpeakText.Request();  req.text = "Hi, what can I help you today?"
            rclpy.spin_until_future_complete(node, spk.call_async(req))

        # 2) start recording & enable hot-word detection

        _call_set_bool(node, "toggle_detection", True)
        time.sleep(1.0)
        #_call_trigger(node, "record_audio")

        node.destroy_node()
        return "awake"

# ──────────── New state ────────────
class ListenForCarryBag(State):
    """Step X – keep recording until transcript contains BOTH 'carry' AND ('bag' OR 'luggage')"""
    def __init__(self):
        super().__init__(["carry_bag_detected"])
    
    def execute(self, bb):
        node = rclpy.create_node("listen_for_carry_bag")
        self.last_transcript = ""
        
        # subscribe to transcripts
        node.create_subscription(
            String, "speech_recognition_transcript",
            lambda m: setattr(self, "last_transcript", m.data.lower()),
            10
        )
        # make sure at least one recording is running
        _call_trigger(node, "record_audio")
        
        node.get_logger().info("Waiting for ‘carry…")
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
            
            lt = self.last_transcript
            if "carry" in lt:
                node.get_logger().info("✅ Detected carry + bag/luggage.")
                
                spk = node.create_client(SpeakText, "/speak_text")
                if _wait_for_service(node, spk, "/speak_text"):
                    req = SpeakText.Request()
                    req.text = "Sure, but can you pass me your luggage, I will help you to hold it"
                    fut = spk.call_async(req)
                    rclpy.spin_until_future_complete(node, fut)
                
                    if fut.result() and fut.result().success:
                        arm_pub = node.create_publisher(Vector3, '/arm_target', 10)
                        arm_msg = Vector3()
                        arm_msg.x = 0.3
                        arm_msg.y = 0.0
                        arm_msg.z = 0.22
                        arm_pub.publish(arm_msg)
                        node.get_logger().info(
                            f"Published arm target: [{arm_msg.x}, {arm_msg.y}, {arm_msg.z}]"
                        )
                
                        time.sleep(10)
                        
                        # 4) call gripper_control to close the gripper
                        if _call_set_bool(node, "gripper_control", True):
                            node.get_logger().info("Gripper close command sent.")
                            time.sleep(2.0)
                            hold_pose = Vector3()
                            hold_pose.x = 0.0
                            hold_pose.y = 0.0
                            hold_pose.z = 0.20
                            arm_pub.publish(hold_pose)
                        else:
                            node.get_logger().error("Failed to call gripper_control.")
                node.destroy_node()
                return "carry_bag_detected"
            
            # didn’t match – clear and record again
            if lt:
                node.get_logger().info(f"Ignoring transcript: {lt}")
                self.last_transcript = ""
                _call_trigger(node, "record_audio")


class WaitAwakeBeforeFollow(State):
    """Step Y – wait for /awake_flag, then trigger a new recording."""
    def __init__(self):
        super().__init__(["awake"])

    def execute(self, bb):
        node = rclpy.create_node("wait_awake_before_follow")
        got_awake = False

        node.create_subscription(
            Int8, "/awake_flag",
            lambda m: setattr(self, "got_awake", True),
            10
        )

        node.get_logger().info("Waiting for /awake_flag (before follow)…")
        while rclpy.ok() and not getattr(self, "got_awake", False):
            rclpy.spin_once(node, timeout_sec=0.2)

        # Once awake-flag arrives, start recording again:
        spk = node.create_client(SpeakText, "/speak_text")
        if _wait_for_service(node, spk, "/speak_text"):
            req = SpeakText.Request()
            req.text = "Hi I am here."
            rclpy.spin_until_future_complete(node, spk.call_async(req))
        #_call_trigger(node, "record_audio")
        node.get_logger().info("Triggered record_audio before ListenForFollow")

        node.destroy_node()
        return "awake"
class ListenForFollow(State):
    """Step 2 – keep recording until transcript contains BOTH ‘follow’ AND ‘me’"""
    def __init__(self):
        super().__init__(["following_started"])

    def execute(self, bb):
        node = rclpy.create_node("listen_for_follow")
        self.last_transcript = ""

        def cb(msg: String):
            self.last_transcript = msg.data.lower()
        node.create_subscription(String, "speech_recognition_transcript", cb, 10)

        # Ensure at least one recording is running
        _call_trigger(node, "record_audio")

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)

            if "follow" in self.last_transcript and "me" in self.last_transcript:
                # Start following
                node.get_logger().info("✅ Detected 'follow me' – following started.")
                spk = node.create_client(SpeakText, "/speak_text")
                if _wait_for_service(node, spk, "/speak_text"):
                    req = SpeakText.Request()
                    req.text = "Ok, I will start following you."
                    rclpy.spin_until_future_complete(node, spk.call_async(req))
                _call_set_bool(node, "/start_following", True)
                node.destroy_node()
                return "following_started"

            # Didn’t match – record again and continue loop
            if self.last_transcript:
                node.get_logger().info(f"Ignoring transcript: {self.last_transcript}")
                self.last_transcript = ""
                _call_trigger(node, "record_audio")


class WaitAwakeStop(State):
    """Step 3 – wait for next /awake_flag, then prompt and record again"""
    def __init__(self):
        super().__init__(["awake"])

    def execute(self, bb):
        node = rclpy.create_node("wait_awake_stop")
        self.awake_recvd = False
        node.create_subscription(Int8, "/awake_flag",
                                 lambda m: setattr(self, "awake_recvd", True), 10)

        node.get_logger().info("Waiting for /awake_flag (stop phase)…")
        while rclpy.ok() and not self.awake_recvd:
            rclpy.spin_once(node, timeout_sec=0.2)

        # greet
        spk = node.create_client(SpeakText, "/speak_text")
        if _wait_for_service(node, spk, "/speak_text"):
            req = SpeakText.Request();  req.text = "Hi, I am here."
            rclpy.spin_until_future_complete(node, spk.call_async(req))

        #_call_trigger(node, "record_audio")
        node.destroy_node()
        return "awake"


class ListenForStop(State):
    """
    Waits for transcript containing 'stop'.
    If found → cancel Nav2 via shell, then always release the bag and finish.
    Otherwise → return 'not_found'.
    """
    def __init__(self):
        super().__init__(["stop", "not_found"])

    def execute(self, bb):
        node = rclpy.create_node("listen_for_stop")
        self.last_transcript = ""

        # keep the mic recording alive
        _call_trigger(node, "record_audio")
        node.create_subscription(
            String, "speech_recognition_transcript",
            lambda m: setattr(self, "last_transcript", m.data.lower()),
            10
        )

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)

            if "stop" in self.last_transcript:
                node.get_logger().info("Stop detected – stopping follow.")

                # 1) Stop following & disable detection
                _call_set_bool(node, "/start_following", False)
                _call_set_bool(node, "toggle_detection", False)

                # 2) Cancel Nav2 goals via shell
                cancel_cmd = [
                    'ros2', 'service', 'call',
                    '/navigate_to_pose/_action/cancel_goal',
                    'action_msgs/srv/CancelGoal',
                    '{"goal_info": {"goal_id": {"uuid": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},'
                    ' "stamp": {"sec": 0, "nanosec": 0}}}'
                ]
                try:
                    subprocess.run(cancel_cmd, check=True)
                    node.get_logger().info("Nav2 goals cancelled via shell command.")
                except subprocess.CalledProcessError as e:
                    node.get_logger().error(f"Failed to cancel via shell: {e}")

                # 3) TTS: “Ok, I will stop following you.”
                spk = node.create_client(SpeakText, "/speak_text")
                if _wait_for_service(node, spk, "/speak_text"):
                    req = SpeakText.Request()
                    req.text = "Ok, I will stop following you."
                    rclpy.spin_until_future_complete(node, spk.call_async(req))

                # 4) Publish arm‐release position
                arm_pub = node.create_publisher(Vector3, '/arm_target', 10)
                release = Vector3(x=0.3, y=0.0, z=0.22)
                arm_pub.publish(release)
                node.get_logger().info(f"Release pose published: {release}")

                # 5) Wait 5 seconds
                node.get_logger().info("Waiting 5 seconds for release pose…")
                time.sleep(5)

                # 6) Open gripper
                if _call_set_bool(node, "gripper_control", False):
                    node.get_logger().info("Gripper open command sent.")
                else:
                    node.get_logger().error("Failed to call gripper_control.")
                time.sleep(5.0)
                # 7) Return arm to home
                home = Vector3(x=0.05, y=0.0, z=0.10)
                arm_pub.publish(home)
                node.get_logger().info(f"Home pose published: {home}")
                
                ac = ActionClient(node, NavigateToPose, 'navigate_to_pose')
                if ac.wait_for_server(timeout_sec=5.0):
                    tf = bb.start_transform  # this is a TransformStamped you stored earlier
                    t = tf.transform.translation
                    #r = tf.transform.rotation

                    goal_msg = NavigateToPose.Goal()
                    # copy the header so the timestamp and frame_id stay correct
                    goal_msg.pose.header = tf.header

                    # wrap Vector3 → Point
                    goal_msg.pose.pose.position = Point(x=t.x, y=t.y, z=t.z)
                    # quaternion can be assigned directly
                    #goal_msg.pose.pose.orientation = r

                    # set your BT XML path
                    goal_msg.behavior_tree = (
                        '/workspaces/isaac_ros-dev/'
                        'src/wheeltec_robot_nav2/param/bt/navigate_to_pose.xml'
                    )

                    send_goal = ac.send_goal_async(goal_msg)
                    rclpy.spin_until_future_complete(node, send_goal)
                    gh = send_goal.result()
                    if gh.accepted:
                        node.get_logger().info("Returning to start pose via Nav2.")
                    else:
                        node.get_logger().error("Start‐pose goal was rejected.")
                else:
                    node.get_logger().error("NavigateToPose action server not available.")     

                node.destroy_node()
                return "stop"

            # Didn’t match → exit early to WAIT_AWAKE_STOP
            if self.last_transcript:
                node.get_logger().info(f"Ignoring transcript: {self.last_transcript}")
                self.last_transcript = ""
                node.destroy_node()
                return "not_found"


# --------------  Build the state-machine --------------------------------------
def main():
    rclpy.init()
    set_ros_loggers()

    bb = Blackboard()          # no shared data needed, but instantiate explicitly
    sm = StateMachine(outcomes=["finished"])
    
    sm.add_state("WAIT_AWAKE_INIT",  WaitAwakePrompt(),{"awake": "LISTEN_CARRY_BAG"})
    sm.add_state("LISTEN_CARRY_BAG", ListenForCarryBag(), {"carry_bag_detected": "WAIT_AWAKE_BEFORE_FOLLOW"})
    sm.add_state("WAIT_AWAKE_BEFORE_FOLLOW", WaitAwakeBeforeFollow(),{"awake": "LISTEN_FOLLOW"})
    sm.add_state("LISTEN_FOLLOW",    ListenForFollow(),   {"following_started": "WAIT_AWAKE_STOP"})
    sm.add_state("WAIT_AWAKE_STOP",  WaitAwakeStop(),     {"awake": "LISTEN_STOP"})
    sm.add_state("LISTEN_STOP",      ListenForStop(),     {"stop": "finished",
                                                           "not_found": "WAIT_AWAKE_STOP"})

    # Optional viewer
    #YasminViewerPub("follow_me_sm", sm)

    try:
        outcome = sm(bb)
        print("State-machine finished with outcome:", outcome)
    except KeyboardInterrupt:
        if sm.is_running():
            sm.cancel_state()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
