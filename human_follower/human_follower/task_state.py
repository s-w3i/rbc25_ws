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
    """Step 1 – trigger on /awake_flag, greet, start recording, enable detection"""
    def __init__(self):
        super().__init__(["awake"])

    def execute(self, bb):
        node             = rclpy.create_node("wait_awake_prompt")
        self.awake_recvd = False

        node.create_subscription(
            Int8, "/awake_flag",
            lambda m: setattr(self, "awake_recvd", True),
            10
        )

        node.get_logger().info("Waiting for /awake_flag …")
        while rclpy.ok() and not self.awake_recvd:
            rclpy.spin_once(node, timeout_sec=0.2)

        # 1) greet
        spk = node.create_client(SpeakText, "/speak_text")
        if _wait_for_service(node, spk, "/speak_text"):
            req = SpeakText.Request();  req.text = "Hi, what can I help you today?"
            rclpy.spin_until_future_complete(node, spk.call_async(req))

        # 2) start recording & enable hot-word detection
        _call_trigger (node, "record_audio")
        _call_set_bool(node, "toggle_detection", True)

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
                _call_set_bool(node, "/start_following", True)
                node.get_logger().info("✅ Detected 'follow me' – following started.")
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

        _call_trigger(node, "record_audio")
        node.destroy_node()
        return "awake"


class ListenForStop(State):
    """
    Waits for transcript containing ‘stop’, ‘follow’, and ‘me’.
    If found → stop following & finish.
    Otherwise → outcome 'not_found' so we loop back to WaitAwakeStop.
    """
    def __init__(self):
        super().__init__(["stop", "not_found"])

    def execute(self, bb):
        node = rclpy.create_node("listen_for_stop")
        self.last_transcript = ""

        node.create_subscription(String, "speech_recognition_transcript",
                                 lambda m: setattr(self, "last_transcript", m.data.lower()), 10)

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)

            if all(w in self.last_transcript for w in ("stop", "follow", "me")):
                # Turn everything off
                _call_set_bool(node, "/start_following", False)
                _call_set_bool(node, "toggle_detection", False)

                spk = node.create_client(SpeakText, "/speak_text")
                if _wait_for_service(node, spk, "/speak_text"):
                    req = SpeakText.Request();  req.text = "Ok, I will stop following you."
                    rclpy.spin_until_future_complete(node, spk.call_async(req))

                node.get_logger().info("✅ Detected 'stop follow me' – finished.")
                node.destroy_node()
                return "stop"

            # Didn’t match – ask again at next /awake_flag
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

    sm.add_state("WAIT_AWAKE_INIT",  WaitAwakePrompt(),   {"awake": "LISTEN_FOLLOW"})
    sm.add_state("LISTEN_FOLLOW",    ListenForFollow(),   {"following_started": "WAIT_AWAKE_STOP"})
    sm.add_state("WAIT_AWAKE_STOP",  WaitAwakeStop(),     {"awake": "LISTEN_STOP"})
    sm.add_state("LISTEN_STOP",      ListenForStop(),     {"stop": "finished",
                                                           "not_found": "WAIT_AWAKE_STOP"})

    # Optional viewer
    YasminViewerPub("follow_me_sm", sm)

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
