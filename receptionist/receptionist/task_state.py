#!/usr/bin/env python3
import time
import json
import os
import rclpy
import cv2
import numpy as np

# Imports from Yasmin for state machine functionality.
from yasmin import State, Blackboard, StateMachine
from yasmin_ros import set_ros_loggers
from yasmin_viewer import YasminViewerPub

# Import required ROS 2 service and message types.
from robot_interfaces.srv import SpeakText, SegmentHumans, DescribeGuest, DescribeGuestSentence, CompareAndFindHost, ComputeGoalPose
from geometry_msgs.msg import PoseStamped, Point, Vector3
from tf_transformations import quaternion_from_euler
from std_srvs.srv import Trigger
from std_msgs.msg import String, Int8
from action_msgs.msg import GoalStatusArray, GoalStatus


from cv_bridge import CvBridge

# Directory where segmented images will be saved.
SEGMENT_DIR = "/home/usern/rbc25_ws/segmented_images"
if not os.path.exists(SEGMENT_DIR):
    os.makedirs(SEGMENT_DIR)


class GuestRegistry:
    """
    A simple class to store guest information.
    Stores guest data in a dictionary keyed by guest name.
    """
    def __init__(self):
        self.guests = {}

    def add_guest(self, name, info):
        """Add or update guest info for a given name."""
        self.guests[name] = info

    def get_guest(self, name):
        """Retrieve guest info by name."""
        return self.guests.get(name)


class SpeakTextState(State):
    """
    State that calls the /speak_text service with a predefined welcome message.
    Runs only once at the start.
    """
    def __init__(self) -> None:
        super().__init__(["speak_done"])

    def execute(self, blackboard: Blackboard) -> str:
        node = rclpy.create_node('speak_text_client')
        client = node.create_client(SpeakText, '/speak_text')
        if not client.wait_for_service(timeout_sec=5.0):
            node.get_logger().error("Service /speak_text not available.")
        else:
            req = SpeakText.Request()
            req.text = "Hello! I will start my receptionist task now"
            future = client.call_async(req)
            rclpy.spin_until_future_complete(node, future)
            if future.result() is not None:
                node.get_logger().info("SpeakText response: " + str(future.result()))
            else:
                node.get_logger().error("Error calling /speak_text.")
        node.destroy_node()
        return "speak_done"


class GoToStartingPointState(State):
    """
    Navigation state: move the robot to the starting point.
    """
    def __init__(self) -> None:
        super().__init__(["nav_done"])
        self._succeeded = False
        self.node = None
        
    def _status_cb(self, msg: GoalStatusArray):
        if not msg.status_list:
            return
        last_status = msg.status_list[-1]
        self.node.get_logger().info(f"Latest status: {last_status.status}")
        if last_status.status == GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().info("Status==SUCCEEDED → exiting loop")
            self._succeeded = True

    def execute(self, blackboard: Blackboard) -> str:
        node = rclpy.create_node('go_to_starting_point')
        self.node = node
        self._succeeded = False
        # subscribe to status
        node.create_subscription(
            GoalStatusArray,
            '/navigate_to_pose/_action/status',
            self._status_cb,
            10
        )
        # publish goal
        pub = node.create_publisher(PoseStamped, '/goal_pose', 10)
        goal = PoseStamped()
        goal.header.stamp = node.get_clock().now().to_msg()
        goal.header.frame_id = "map"
        goal.pose.position.x = 4.70  
        goal.pose.position.y =  -0.84
        goal.pose.position.z = 0.0
        yaw = 1.57  # e.g. 90° facing “north” in your map

        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw
        pub.publish(goal)
        node.get_logger().info("Published starting‐point goal, waiting for result…")

        # wait until we see status == 4
        # while rclpy.ok() and not self._succeeded:
        #     rclpy.spin_once(node, timeout_sec=0.1)

        node.get_logger().info("Navigation succeeded (status=4).")

        spk2 = node.create_client(SpeakText, '/speak_text')
        if spk2.wait_for_service(timeout_sec=5.0):
            req2 = SpeakText.Request()
            req2.text = "Ready to accept guest, please call me if you need assistant."
            fut2 = spk2.call_async(req2)
            rclpy.spin_until_future_complete(node, fut2)
        else:
            node.get_logger().error("Service /speak_text not available to announce arrival.")

        node.get_logger().info("Navigation succeeded (status=4).")
        node.destroy_node()

        return "nav_done"

class StartConversationState(State):
    """
    State that waits for the /awake_flag topic before triggering /start_conversation.
    Guest info is then received from /guest_info and stored in:
      - The global GuestRegistry.
      - The blackboard field "current_guest" (for introduction).
    Also increments a turn counter ("turn") on the blackboard.
    """
    def __init__(self) -> None:
        super().__init__(["conversation_done"])
        self.guest_info_received = False
        self.guest_info = None
        self.awake_received = False

    def guest_info_callback(self, msg: String):
        try:
            self.guest_info = json.loads(msg.data)
            self.guest_info_received = True
        except Exception as e:
            self.get_logger().error("Error parsing guest info: " + str(e))

    def awake_callback(self, msg: Int8):
        self.awake_received = True

    def execute(self, blackboard: Blackboard) -> str:
        self.guest_info = None
        self.guest_info_received = False
        self.awake_received = False

        node = rclpy.create_node('start_conversation_node')

        # Wait for /awake_flag.
        awake_sub = node.create_subscription(Int8, '/awake_flag', self.awake_callback, 10)
        node.get_logger().info("Waiting for awake flag on /awake_flag...")
        while rclpy.ok() and not self.awake_received:
            rclpy.spin_once(node, timeout_sec=1.0)
        node.get_logger().info("Awake flag received.")

        # Trigger /start_conversation service.
        client = node.create_client(Trigger, '/start_conversation')
        if not client.wait_for_service(timeout_sec=5.0):
            node.get_logger().error("Service /start_conversation not available.")
        else:
            req = Trigger.Request()
            future = client.call_async(req)
            rclpy.spin_until_future_complete(node, future)
            if future.result() is not None:
                node.get_logger().info("Trigger response: " + str(future.result()))
            else:
                node.get_logger().error("Error calling /start_conversation.")

        # Subscribe to /guest_info.
        guest_sub = node.create_subscription(String, '/guest_info', self.guest_info_callback, 10)
        node.get_logger().info("Waiting for guest info on /guest_info...")
        while rclpy.ok() and not self.guest_info_received:
            rclpy.spin_once(node, timeout_sec=1.0)

        if self.guest_info_received:
            if "guest_registry" in blackboard:
                guest_registry = blackboard["guest_registry"]
            else:
                node.get_logger().warn("Guest registry not found in blackboard; creating new registry.")
                guest_registry = GuestRegistry()
                blackboard["guest_registry"] = guest_registry
            guest_name = self.guest_info.get("name", "unknown")
            guest_registry.add_guest(guest_name, self.guest_info)
            node.get_logger().info(f"Stored guest info for {guest_name}: {self.guest_info}")
            blackboard["current_guest"] = self.guest_info
            if "turn" in blackboard:
                blackboard["turn"] += 1
            else:
                blackboard["turn"] = 1
            node.get_logger().info(f"Turn count is now {blackboard['turn']}.")
        node.destroy_node()
        return "conversation_done"
    
class RegisterGuestState(State):
    """
    Calls /segment_humans, saves the image, then calls /describe_guest
    to enrich guest_info with height_cm, age, etc.
    """
    def __init__(self) -> None:
        super().__init__(["registered"])
        self.bridge = CvBridge()

    def execute(self, blackboard: Blackboard) -> str:
        node = rclpy.create_node('register_guest_node')
        # Prompt guest
        spk1 = node.create_client(SpeakText, '/speak_text')
        if spk1.wait_for_service(timeout_sec=5.0):
            req1 = SpeakText.Request()
            req1.text = "Taking your picture to register you into the system."
            fut1 = spk1.call_async(req1)
            rclpy.spin_until_future_complete(node, fut1)

        # Call segmentation
        seg_client = node.create_client(SegmentHumans, '/segment_humans')
        if not seg_client.wait_for_service(timeout_sec=5.0):
            node.get_logger().error("Segmentation service not available.")
            node.destroy_node()
            return "registered"

        attempt = 0
        while True:
            attempt += 1
            fut_seg = seg_client.call_async(SegmentHumans.Request())
            rclpy.spin_until_future_complete(node, fut_seg)
            res = fut_seg.result()
            if not res or not res.success:
                node.get_logger().error(f"Segmentation failed (attempt {attempt}). Retry.")
                continue
            if len(res.segmented_images) != 1:
                # Retake prompt
                if spk1.wait_for_service(timeout_sec=5.0):
                    req_re = SpeakText.Request()
                    req_re.text = "Retaking the picture. Please hold still."
                    fut_re = spk1.call_async(req_re)
                    rclpy.spin_until_future_complete(node, fut_re)
                time.sleep(1.0)
                continue

            # Convert & save
            try:
                cv_img = self.bridge.imgmsg_to_cv2(res.segmented_images[0], desired_encoding="bgr8")
            except Exception as e:
                node.get_logger().error(f"Image conversion error: {e}")
                continue

            guest_info = blackboard["current_guest"]
            name = guest_info.get("name", "unknown")
            filepath = os.path.join(SEGMENT_DIR, f"{name}.jpg")
            cv2.imwrite(filepath, cv_img)
            node.get_logger().info(f"Saved image: {filepath}")

            # Store centroids
            guest_info["centroids"] = [
                {"x": p.x, "y": p.y, "z": p.z} for p in res.centroids
            ]
            guest_info["segmented_file"] = filepath

            # Call describe_guest
            desc_cli = node.create_client(DescribeGuest, '/describe_guest')
            if desc_cli.wait_for_service(timeout_sec=5.0):
                req_d = DescribeGuest.Request()
                req_d.image_path = filepath
                fut_d = desc_cli.call_async(req_d)
                rclpy.spin_until_future_complete(node, fut_d)
                dres = fut_d.result()
                if dres and dres.success:
                    try:
                        parsed = json.loads(dres.description)
                        guest_info.update(parsed)
                        node.get_logger().info(f"Description added: {parsed}")
                    except Exception as e:
                        node.get_logger().error(f"Failed to parse description JSON: {e}")
                else:
                    node.get_logger().error(f"DescribeGuest failed: {getattr(dres,'message','none')}")

            else:
                node.get_logger().error("Service /describe_guest not available.")

            # Update registry
            blackboard["guest_registry"].add_guest(name, guest_info)

            # Final announce
            spk2 = node.create_client(SpeakText, '/speak_text')
            if spk2.wait_for_service(timeout_sec=5.0):
                req2 = SpeakText.Request()
                req2.text = f"Hi {name}, I have registered you successfully. Please follow me"
                fut2 = spk2.call_async(req2)
                rclpy.spin_until_future_complete(node, fut2)

            break  # exit loop

        node.destroy_node()
        return "registered"


class CheckTurnState(State):
    """
    State that checks the turn counter on the blackboard.
    Returns "first" if turn == 1, "second" if turn == 2.
    """
    def __init__(self) -> None:
        super().__init__(["first", "second"])

    def execute(self, blackboard: Blackboard) -> str:
        turn = blackboard["turn"] if "turn" in blackboard else 0
        return "first" if turn == 1 else "second"

class GoToHostFirstState(State):
    """
    For first guest session: navigation state to host.
    """
    def __init__(self) -> None:
        super().__init__(["nav_done"])
        self._succeeded = False
        self.node = None
        
    def _status_cb(self, msg: GoalStatusArray):
        if not msg.status_list:
            return
        last_status = msg.status_list[-1]
        self.node.get_logger().info(f"Latest status: {last_status.status}")
        if last_status.status == GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().info("Status==SUCCEEDED → exiting loop")
            self._succeeded = True

    def execute(self, blackboard: Blackboard) -> str:
        node = rclpy.create_node('go_to_starting_point')
        self.node = node
        self._succeeded = False
        # subscribe to status
        node.create_subscription(
            GoalStatusArray,
            '/navigate_to_pose/_action/status',
            self._status_cb,
            10
        )
        # publish goal
        pub = node.create_publisher(PoseStamped, '/goal_pose', 10)
        goal = PoseStamped()
        goal.header.stamp = node.get_clock().now().to_msg()
        goal.header.frame_id = "map"
        goal.pose.position.x = 8.6  
        goal.pose.position.y =  -6.16
        goal.pose.position.z = 0.0
        yaw = 3.14159 # e.g. 180° facing south in your map

        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw
        pub.publish(goal)
        node.get_logger().info("Published starting‐point goal, waiting for result…")

        # wait until we see status == 4
        # while rclpy.ok() and not self._succeeded:
        #     rclpy.spin_once(node, timeout_sec=0.1)

        node.get_logger().info("Navigation succeeded (status=4).")
        node.destroy_node()
        return "nav_done"

class SegmentHostState(State):
    """
    After moving to host, call /segment_humans until exactly one person is segmented.
    Save the image as 'host.jpg' and store centroids on the blackboard.
    """
    def __init__(self) -> None:
        super().__init__(outcomes=["segmented"])
        # Placeholder for our ROS2 node and success flag
        self.node = None
        self._succeeded = False
        self.bridge = CvBridge()

    def _status_cb(self, msg: GoalStatusArray):
        if not msg.status_list:
            return
        last_status = msg.status_list[-1]
        # Use the stored node for logging
        self.node.get_logger().info(f"Latest status: {last_status.status}")
        if last_status.status == GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().info("Status==SUCCEEDED → exiting loop")
            self._succeeded = True

    def execute(self, blackboard: Blackboard) -> str:
        # Initialize our node
        self.node = rclpy.create_node('segment_host_node')
        node = self.node
        self._succeeded = False

        # Wait for segmentation service
        client = node.create_client(SegmentHumans, '/segment_humans')
        if not client.wait_for_service(timeout_sec=5.0):
            node.get_logger().error("Service /segment_humans not available.")
            node.destroy_node()
            return "segmented"

        # Retry until exactly one person is segmented
        while rclpy.ok():
            req = SegmentHumans.Request()
            future = client.call_async(req)
            rclpy.spin_until_future_complete(node, future)
            res = future.result()

            if res and res.success and len(res.segmented_images) == 1:
                cv_img = self.bridge.imgmsg_to_cv2(
                    res.segmented_images[0], desired_encoding="bgr8"
                )
                filepath = os.path.join(SEGMENT_DIR, "host.jpg")
                cv2.imwrite(filepath, cv_img)
                node.get_logger().info(f"Saved host image: {filepath}")

                # Store results on the blackboard
                blackboard["host_image"] = filepath
                blackboard["host_centroids"] = [
                    {"x": p.x, "y": p.y, "z": p.z} for p in res.centroids
                ]
                break
            else:
                node.get_logger().warn(
                    f"Segmentation returned {len(res.segmented_images) if res else 'no'} images, retrying..."
                )
                time.sleep(1.0)

        # Compute a goal pose to approach the host
        cmp_cli = node.create_client(ComputeGoalPose, '/compute_goal_pose')
        if cmp_cli.wait_for_service(timeout_sec=5.0):
            cmp_req = ComputeGoalPose.Request()
            cmp_req.target_frame = "person_1"
            cmp_fut = cmp_cli.call_async(cmp_req)
            rclpy.spin_until_future_complete(node, cmp_fut)
            cmp_res = cmp_fut.result()

            if cmp_res and cmp_res.success:
                blackboard["host_goal_pose"] = cmp_res.goal_pose
                node.get_logger().info(f"Computed goal_pose: {cmp_res.goal_pose}")
            else:
                node.get_logger().error(
                    f"ComputeGoalPose failed: {getattr(cmp_res, 'message', '<no response>')}"
                )
        else:
            node.get_logger().error("Service /compute_goal_pose not available.")

        # Subscribe to the navigation status and wait until succeeded
        node.create_subscription(
            GoalStatusArray,
            '/navigate_to_pose/_action/status',
            self._status_cb,
            10
        )
        node.get_logger().info("Waiting for navigation result…")
        # while rclpy.ok() and not self._succeeded:
        #     rclpy.spin_once(node, timeout_sec=0.1)

        node.get_logger().info("Host navigation succeeded (status=4).")
        node.destroy_node()
        return "segmented"

class CompareAndFindHostState(State):
    """
    For the second turn: segment humans once, then call /compare_and_find_host
    to pick the best_person from the segmented results.
    """
    def __init__(self) -> None:
        super().__init__(outcomes=["compared"])
        # Placeholder for our ROS2 node and success flag
        self.node = None
        self._succeeded = False
        self.bridge = CvBridge()

    def _status_cb(self, msg: GoalStatusArray):
        if not msg.status_list:
            return
        last_status = msg.status_list[-1]
        # Use the stored node for logging
        self.node.get_logger().info(f"Latest status: {last_status.status}")
        if last_status.status == GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().info("Status==SUCCEEDED → exiting loop")
            self._succeeded = True

    def execute(self, blackboard: Blackboard) -> str:
        self.node = rclpy.create_node('compare_and_find_host_node')
        node = self.node
        self._succeeded = False
        # 1. Call segmentation once
        seg_cli = node.create_client(SegmentHumans, '/segment_humans')
        if not seg_cli.wait_for_service(timeout_sec=5.0):
            node.get_logger().error("Service /segment_humans not available.")
            node.destroy_node()
            return "compared"
        fut_seg = seg_cli.call_async(SegmentHumans.Request())
        rclpy.spin_until_future_complete(node, fut_seg)
        seg_res = fut_seg.result()
        if not seg_res or not seg_res.success:
            node.get_logger().error("Segmentation failed.")
            node.destroy_node()
            return "compared"

        # 2. Call CompareAndFindHost
        cmp_cli = node.create_client(CompareAndFindHost, 'compare_and_find_host')
        if not cmp_cli.wait_for_service(timeout_sec=5.0):
            node.get_logger().error("Service compare_and_find_host not available.")
            node.destroy_node()
            return "compared"

        req_cmp = CompareAndFindHost.Request()
        # pass the stored host.jpg path
        req_cmp.host_image_dir = blackboard["host_image"]
        # pass all segmented images & centroids
        req_cmp.segmented_images = seg_res.segmented_images
        req_cmp.centroids        = seg_res.centroids

        fut_cmp = cmp_cli.call_async(req_cmp)
        rclpy.spin_until_future_complete(node, fut_cmp)
        cmp_res = fut_cmp.result()
        if cmp_res and cmp_res.success:
            blackboard["best_person"] = cmp_res.best_person
            node.get_logger().info(f"Best person is {cmp_res.best_person}")
            best_fit = cmp_res.best_person
        else:
            node.get_logger().error("compare_and_find_host failed or returned no best_person.")
        
        cmp_cli = node.create_client(ComputeGoalPose, '/compute_goal_pose')
        if cmp_cli.wait_for_service(timeout_sec=5.0):
            cmp_req = ComputeGoalPose.Request()
            cmp_req.target_frame = best_fit
            cmp_fut = cmp_cli.call_async(cmp_req)
            rclpy.spin_until_future_complete(node, cmp_fut)
            cmp_res = cmp_fut.result()

            if cmp_res and cmp_res.success:
                blackboard["host_goal_pose"] = cmp_res.goal_pose
                node.get_logger().info(f"Computed goal_pose: {cmp_res.goal_pose}")
            else:
                node.get_logger().error(
                    f"ComputeGoalPose failed: {getattr(cmp_res, 'message', '<no response>')}"
                )
        else:
            node.get_logger().error("Service /compute_goal_pose not available.")

        # Subscribe to the navigation status and wait until succeeded
        node.create_subscription(
            GoalStatusArray,
            '/navigate_to_pose/_action/status',
            self._status_cb,
            10
        )
        node.get_logger().info("Waiting for navigation result…")
        # while rclpy.ok() and not self._succeeded:
        #     rclpy.spin_once(node, timeout_sec=0.1)

        node.get_logger().info("Host navigation succeeded (status=4).")

        node.destroy_node()
        return "compared"

class GoToHostSecondState(State):
    """
    For second guest session: navigation state to host.
    """
    def __init__(self) -> None:
        super().__init__(["nav_done"])
        self._succeeded = False
        self.node = None        
        
    def _status_cb(self, msg: GoalStatusArray):
        if not msg.status_list:
            return
        last_status = msg.status_list[-1]
        self.node.get_logger().info(f"Latest status: {last_status.status}")
        if last_status.status == GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().info("Status==SUCCEEDED → exiting loop")
            self._succeeded = True

    def execute(self, blackboard: Blackboard) -> str:
        node = rclpy.create_node('go_to_starting_point')
        self.node = node
        self._succeeded = False
        # subscribe to status
        node.create_subscription(
            GoalStatusArray,
            '/navigate_to_pose/_action/status',
            self._status_cb,
            10
        )
        # publish goal
        pub = node.create_publisher(PoseStamped, '/goal_pose', 10)
        goal = PoseStamped()
        goal.header.stamp = node.get_clock().now().to_msg()
        goal.header.frame_id = "map"
        goal.pose.position.x = 8.6  
        goal.pose.position.y =  -6.16
        goal.pose.position.z = 0.0
        yaw = 3.14159  # e.g. 180° facing “south” in your map

        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw
        pub.publish(goal)
        node.get_logger().info("Published starting‐point goal, waiting for result…")

        # wait until we see status == 4
        # while rclpy.ok() and not self._succeeded:
        #     rclpy.spin_once(node, timeout_sec=0.1)

        node.get_logger().info("Navigation succeeded (status=4).")
        node.destroy_node()
        return "nav_done"

class IntroduceGuestFirstState(State):
    """
    For first guest session: introduces the current (first) guest to the host.
    """
    def __init__(self) -> None:
        super().__init__(["introduced"])

    def execute(self, blackboard: Blackboard) -> str:
        node = rclpy.create_node('introduce_guest_first')
        if "current_guest" not in blackboard:
            node.get_logger().error("No current guest info available!")
            node.destroy_node()
            return "introduced"
        current_guest = blackboard["current_guest"]
        HOST_NAME = "Max"
        guest_name = current_guest.get("name", "unknown")
        guest_drink = current_guest.get("drink", "unknown")
        text = f"Hi {HOST_NAME}, this is {guest_name}, His favourite drink is {guest_drink}."
        node.get_logger().info("Introduce Guest (first session) text: " + text)
        client = node.create_client(SpeakText, '/speak_text')
        if not client.wait_for_service(timeout_sec=5.0):
            node.get_logger().error("Service /speak_text not available (first session).")
        else:
            req = SpeakText.Request()
            req.text = text
            future = client.call_async(req)
            rclpy.spin_until_future_complete(node, future)
            if future.result() is not None:
                node.get_logger().info("Introduce guest (first session) response: " + str(future.result()))
            else:
                node.get_logger().error("Error calling /speak_text (first session).")
        node.destroy_node()
        return "introduced"

class IntroduceGuestSecondState(State):
    """
    For second guest session: introduces the current (second) guest to the host.
    """
    def __init__(self) -> None:
        super().__init__(["introduced"])

    def execute(self, blackboard: Blackboard) -> str:
        node = rclpy.create_node('introduce_guest_second')
        if "current_guest" not in blackboard:
            node.get_logger().error("No current guest info available!")
            node.destroy_node()
            return "introduced"
        current_guest = blackboard["current_guest"]
        HOST_NAME = "Max"
        guest_name = current_guest.get("name", "unknown")
        guest_drink = current_guest.get("drink", "unknown")
        text = f"Hi {HOST_NAME}, this is {guest_name}, His favourite drink is {guest_drink}."
        node.get_logger().info("Introduce Guest (second session) text: " + text)
        client = node.create_client(SpeakText, '/speak_text')
        if not client.wait_for_service(timeout_sec=5.0):
            node.get_logger().error("Service /speak_text not available (second session).")
        else:
            req = SpeakText.Request()
            req.text = text
            future = client.call_async(req)
            rclpy.spin_until_future_complete(node, future)
            if future.result() is not None:
                node.get_logger().info("Introduce guest (second session) response: " + str(future.result()))
            else:
                node.get_logger().error("Error calling /speak_text (second session).")
        node.destroy_node()
        return "introduced"

class IntroduceFirstGuestState(State):
    """
    For second guest session: introduces the first guest to the current (second) guest
    by calling /describe_guest_sentence for the first guest, then tacking on
    “Hi <second_guest>, please meet <first_guest>, ” before speaking.
    """
    def __init__(self) -> None:
        super().__init__(["introduced"])

    def execute(self, blackboard: Blackboard) -> str:
        node = rclpy.create_node('introduce_first_guest')
        if "guest_registry" not in blackboard or "current_guest" not in blackboard:
            node.get_logger().error("Guest registry or current guest info not available!")
            node.destroy_node()
            return "introduced"

        guest_registry = blackboard["guest_registry"]
        current_guest = blackboard["current_guest"]
        second_name = current_guest.get("name", "unknown")

        # Find the first guest (not the current one)
        first_guest = None
        for name, info in guest_registry.guests.items():
            if name != second_name:
                first_guest = info
                break

        if first_guest is None:
            node.get_logger().info("No first guest info available to introduce.")
            node.destroy_node()
            return "introduced"

        first_name = first_guest.get("name", "unknown")

        # 1) Call describe_guest_sentence service
        raw_sentence = None
        desc_cli = node.create_client(DescribeGuestSentence, '/describe_guest_sentence')
        if desc_cli.wait_for_service(timeout_sec=5.0):
            req = DescribeGuestSentence.Request()
            req.guest_info_json = json.dumps(first_guest, ensure_ascii=False)
            fut = desc_cli.call_async(req)
            rclpy.spin_until_future_complete(node, fut)
            res = fut.result()
            if res and res.success:
                raw_sentence = res.description  # e.g. "He is a 23-year-old male..."
                node.get_logger().info(f"describe_guest_sentence returned: {raw_sentence}")
            else:
                node.get_logger().error(f"describe_guest_sentence failed: {getattr(res, 'message', '')}")
        else:
            node.get_logger().error("/describe_guest_sentence service not available.")

        # 2) Build the full intro sentence
        if raw_sentence:
            # Ensure it ends without a period so we can comma-concatenate cleanly
            raw_sentence = raw_sentence.rstrip('.')
            full_text = f"Hi {second_name}, please meet {first_name}, {raw_sentence}."
        else:
            # Fallback
            full_text = f"Hi {second_name}, please meet {first_name}."

        node.get_logger().info(f"Final introduction text: {full_text}")

        # 3) Send to /speak_text
        spk = node.create_client(SpeakText, '/speak_text')
        if spk.wait_for_service(timeout_sec=5.0):
            spk_req = SpeakText.Request()
            spk_req.text = full_text
            spk_fut = spk.call_async(spk_req)
            rclpy.spin_until_future_complete(node, spk_fut)
            if spk_fut.result() is not None:
                node.get_logger().info("Spoken introduction complete.")
            else:
                node.get_logger().error("Error calling /speak_text for introduction.")
        else:
            node.get_logger().error("/speak_text service not available for introduction.")

        node.destroy_node()
        return "introduced"

class GoToChairDetectionState(State):
    """
    Navigation state: move the robot to the chair‐detection area
    before running the empty‐chair detector.
    """
    def __init__(self) -> None:
        super().__init__(["nav_done"])
        self._succeeded = False
        self.node = None

    def _status_cb(self, msg: GoalStatusArray):
        if not msg.status_list:
            return
        last_status = msg.status_list[-1]
        self.node.get_logger().info(f"[GoToChairDetection] status: {last_status.status}")
        if last_status.status == GoalStatus.STATUS_SUCCEEDED:
            self._succeeded = True

    def execute(self, blackboard: Blackboard) -> str:
        node = rclpy.create_node('go_to_chair_detection')
        self.node = node
        self._succeeded = False

        # 1) subscribe to Nav2 status
        node.create_subscription(
            GoalStatusArray,
            '/navigate_to_pose/_action/status',
            self._status_cb,
            10
        )

        # 2) publish the detection‐area goal
        pub = node.create_publisher(PoseStamped, '/goal_pose', 10)
        goal = PoseStamped()
        goal.header.stamp = node.get_clock().now().to_msg()
        goal.header.frame_id = "map"
        # ← fill in your chair‐detection coordinates:
        goal.header.frame_id = "map"
        goal.pose.position.x = 8.6  
        goal.pose.position.y =  -6.16
        goal.pose.position.z = 0.0
        yaw = 3.14159
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw
        pub.publish(goal)
        node.get_logger().info("→ Moving to chair‐detection area…")

        # 3) wait for success
        # while rclpy.ok() and not self._succeeded:
        #     rclpy.spin_once(node, timeout_sec=0.1)

        node.get_logger().info("→ Arrived at detection area.")
        node.destroy_node()
        return "nav_done"

class EmptyChairDetectionState(State):
    """
    State that:
      1. Announces via /speak_text: "Hi <guest_name>, I will find you an empty seat".
      2. Calls the empty chair detection service 'detect_empty_chair'.
      3. If detection succeeds, announces: "Hi <guest_name>, you may have your seat here".
      4. For the second guest session, after seat assignment, calls /speak_text one final time to announce "All tasks have been completed".
      5. Returns "loop" if turn < 2, otherwise "finished".
    """
    def __init__(self) -> None:
        super().__init__(["loop", "finished"])

        self.node = None
        self._succeeded = False
        self.bridge = CvBridge()

    def _status_cb(self, msg: GoalStatusArray):
        if not msg.status_list:
            return
        last_status = msg.status_list[-1]
        # Use the stored node for logging
        self.node.get_logger().info(f"Latest status: {last_status.status}")
        if last_status.status == GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().info("Status==SUCCEEDED → exiting loop")
            self._succeeded = True

    def execute(self, blackboard: Blackboard) -> str:
        node = rclpy.create_node('empty_chair_detection')
        self._succeeded = False
        if "current_guest" not in blackboard:
            node.get_logger().error("No current guest info available!")
            node.destroy_node()
            return "loop"
        current_guest = blackboard["current_guest"]
        guest_name = current_guest.get("name", "unknown")
        
        # 1. Announce empty seat search.
        speak_client = node.create_client(SpeakText, '/speak_text')
        if not speak_client.wait_for_service(timeout_sec=5.0):
            node.get_logger().error("Service /speak_text not available for empty seat announcement.")
        else:
            req = SpeakText.Request()
            req.text = f"Hi {guest_name}, I will find you an empty seat"
            future = speak_client.call_async(req)
            rclpy.spin_until_future_complete(node, future)
            if future.result() is None or not future.result().success:
                node.get_logger().error("Error in speak_text for empty seat announcement.")

        detect_client = node.create_client(Trigger, 'detect_empty_chair')
        if not detect_client.wait_for_service(timeout_sec=5.0):
            node.get_logger().error("Service detect_empty_chair not available.")
        else:
            req = Trigger.Request()
            future = detect_client.call_async(req)
            rclpy.spin_until_future_complete(node, future)
            if future.result() is not None and future.result().success:
                node.get_logger().info("Empty chair detected: " + str(future.result().message))

        cmp_cli = node.create_client(ComputeGoalPose, '/compute_goal_pose')
        if cmp_cli.wait_for_service(timeout_sec=5.0):
            cmp_req = ComputeGoalPose.Request()
            cmp_req.target_frame = "empty_chair"
            cmp_fut = cmp_cli.call_async(cmp_req)
            rclpy.spin_until_future_complete(node, cmp_fut)
            cmp_res = cmp_fut.result()

            if cmp_res and cmp_res.success:
                blackboard["host_goal_pose"] = cmp_res.goal_pose
                node.get_logger().info(f"Computed goal_pose: {cmp_res.goal_pose}")
            else:
                node.get_logger().error(
                    f"ComputeGoalPose failed: {getattr(cmp_res, 'message', '<no response>')}"
                )
        else:
            node.get_logger().error("Service /compute_goal_pose not available.")

        # Subscribe to the navigation status and wait until succeeded
        node.create_subscription(
            GoalStatusArray,
            '/navigate_to_pose/_action/status',
            self._status_cb,
            10
        )
        node.get_logger().info("Waiting for navigation result…")
        # while rclpy.ok() and not self._succeeded:
        #     rclpy.spin_once(node, timeout_sec=0.1)

        node.get_logger().info("chair navigation succeeded (status=4).")

        time.sleep(5)
        
        # 2. Call the empty chair detection service.
        detect_client = node.create_client(Trigger, 'point_empty_chair')
        if not detect_client.wait_for_service(timeout_sec=5.0):
            node.get_logger().error("Service point_empty_chair not available.")
        else:
            req = Trigger.Request()
            future = detect_client.call_async(req)
            rclpy.spin_until_future_complete(node, future)
            if future.result() is not None and future.result().success:
                node.get_logger().info("Empty chair detected: " + str(future.result().message))
                # 3. Announce seat assignment.
                assign_client = node.create_client(SpeakText, '/speak_text')
                if not assign_client.wait_for_service(timeout_sec=5.0):
                    node.get_logger().error("Service /speak_text not available for seat assignment.")
                else:
                    req = SpeakText.Request()
                    req.text = f"Hi {guest_name}, you may have your seat here"
                    future_assign = assign_client.call_async(req)
                    rclpy.spin_until_future_complete(node, future_assign)
                    if future_assign.result() is not None:
                        node.get_logger().info("Seat assignment speak response: " + str(future_assign.result()))
                    else:
                        node.get_logger().error("Error calling /speak_text for seat assignment")
            else:
                node.get_logger().error("No empty chair detected or detection service failed.")
        
        arm_pub = node.create_publisher(Vector3, '/arm_target', 10)
        arm_msg = Vector3()
        arm_msg.x = 0.05
        arm_msg.y = 0.0
        arm_msg.z = 0.1
        arm_pub.publish(arm_msg)
        node.get_logger().info("Published arm target at /arm_target: [0.05, 0.0, 0.1]")

        time.sleep(10)
        # 4. Check turn counter.
        current_turn = blackboard["turn"] if "turn" in blackboard else 0
        if current_turn >= 2:
            final_client = node.create_client(SpeakText, '/speak_text')
            if not final_client.wait_for_service(timeout_sec=5.0):
                node.get_logger().error("Service /speak_text not available for final message.")
            else:
                req = SpeakText.Request()
                req.text = "All tasks have been completed"
                future_final = final_client.call_async(req)
                rclpy.spin_until_future_complete(node, future_final)
                if future_final.result() is not None:
                    node.get_logger().info("Final speak response: " + str(future_final.result()))
                else:
                    node.get_logger().error("Error calling /speak_text for final message.")
            node.destroy_node()
            return "finished"
        else:
            node.get_logger().info("Looping for next guest. Current turn: " + str(current_turn))
            node.destroy_node()
            return "loop"

def main():
    rclpy.init()
    set_ros_loggers()

    # Create an explicit blackboard and initialize global storage.
    blackboard = Blackboard()
    blackboard["guest_registry"] = GuestRegistry()
    blackboard["turn"] = 0

    # Create the state machine with final outcome "finished".
    sm = StateMachine(outcomes=["finished"])

    # Define state transitions.
    sm.add_state("SPEAK", SpeakTextState(), transitions={"speak_done": "GO_STARTING_POINT"})
    sm.add_state("GO_STARTING_POINT", GoToStartingPointState(), transitions={"nav_done": "CONVERSATION"})
    sm.add_state("CONVERSATION", StartConversationState(), transitions={"conversation_done": "REGISTER_GUEST"})
    sm.add_state("REGISTER_GUEST", RegisterGuestState(), transitions={"registered": "CHECK_TURN"})
    sm.add_state("CHECK_TURN", CheckTurnState(), transitions={"first": "GO_TO_HOST_FIRST", "second": "GO_TO_HOST_SECOND"})
    # For first guest session.
    sm.add_state("GO_TO_HOST_FIRST", GoToHostFirstState(), transitions={"nav_done": "SEGMENT_HOST"})
    sm.add_state("SEGMENT_HOST",SegmentHostState(),transitions={"segmented": "INTRODUCE_GUEST_FIRST"})
    sm.add_state("INTRODUCE_GUEST_FIRST", IntroduceGuestFirstState(), transitions={"introduced": "GO_TO_DETECTION"})
    sm.add_state("GO_TO_DETECTION",GoToChairDetectionState(),transitions={"nav_done": "EMPTY_CHAIR"})
    # For second guest session.
    sm.add_state("GO_TO_HOST_SECOND", GoToHostSecondState(), transitions={"nav_done": "COMPARE_AND_FIND_HOST"})
    sm.add_state("COMPARE_AND_FIND_HOST", CompareAndFindHostState(), transitions={"compared": "INTRODUCE_GUEST_SECOND"})
    sm.add_state("INTRODUCE_GUEST_SECOND", IntroduceGuestSecondState(), transitions={"introduced": "GO_TO_DETECTION_FOR_INTRO"})
    sm.add_state("GO_TO_DETECTION_FOR_INTRO", GoToChairDetectionState(), transitions={"nav_done": "INTRODUCE_FIRST_GUEST"})
    sm.add_state("INTRODUCE_FIRST_GUEST", IntroduceFirstGuestState(), transitions={"introduced": "EMPTY_CHAIR"})
    # Common empty chair detection.
    sm.add_state("EMPTY_CHAIR", EmptyChairDetectionState(), transitions={"loop": "GO_STARTING_POINT",
                                                                          "finished": "finished"})

    YasminViewerPub("state_machine_demo", sm)

    try:
        outcome = sm(blackboard)
        print("State machine finished with outcome:", outcome)
    except KeyboardInterrupt:
        if sm.is_running():
            sm.cancel_state()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
