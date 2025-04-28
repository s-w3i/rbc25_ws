#!/usr/bin/env python3
"""
people_follower_goal.py

Listens to /people_transform (TransformStamped) for the frame "people_1",
and publishes a Nav2 goal when start_following service is ON.
When stop following, it cancels the current navigation goal.
"""

import math
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, TransformStamped
from std_srvs.srv import SetBool
from action_msgs.srv import CancelGoal
from action_msgs.msg import GoalInfo
import tf2_ros
from tf_transformations import quaternion_from_euler  # sudo apt install python3-tf-transformations


class PeopleFollowerGoal(Node):
    def __init__(self):
        super().__init__('people_follower_goal')

        # Frames
        self.global_frame = 'map'
        self.base_frame   = 'base_footprint'

        # TF buffer / listener
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribe to people transforms
        self.create_subscription(
            TransformStamped,
            '/people_transform',
            self.people_cb,
            10
        )

        # Publisher to Nav2
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # Client to cancel goals
        self.cancel_goal_client = self.create_client(CancelGoal, '/navigate_to_pose/_action/cancel_goal')
        while not self.cancel_goal_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for /navigate_to_pose/_action/cancel_goal service...')

        # Service to start/stop following
        self.following = False
        self.create_service(SetBool, '/start_following', self.start_following_cb)

        self.get_logger().info("PeopleFollowerGoal node started (waiting for /start_following service call)")

    def start_following_cb(self, request, response):
        """Callback to start or stop following based on service call."""
        self.following = request.data
        if self.following:
            self.get_logger().info("Following ENABLED")
            response.message = "Following started."
        else:
            self.get_logger().info("Following DISABLED, cancelling goal...")
            self.cancel_current_goal()
            response.message = "Following stopped and goal cancelled."
        response.success = True
        return response

    def cancel_current_goal(self):
        """Cancel the current navigation goal."""
        cancel_req = CancelGoal.Request()
        # Empty GoalInfo (zero UUID cancels all active goals)
        cancel_req.goal_info = GoalInfo()

        future = self.cancel_goal_client.call_async(cancel_req)

        def _cancel_done(fut):
            try:
                result = fut.result()
                if result.return_code == 0:
                    self.get_logger().info("Successfully cancelled the current goal.")
                else:
                    self.get_logger().warn(f"Failed to cancel goal. Return code: {result.return_code}")
            except Exception as e:
                self.get_logger().error(f"Exception while cancelling goal: {e}")

        future.add_done_callback(_cancel_done)

    def people_cb(self, msg: TransformStamped):
        """Publish goal only if following is enabled."""
        if not self.following:
            return
        if msg.child_frame_id != 'people_1':
            return

        # Extract person’s position in map frame
        px = msg.transform.translation.x
        py = msg.transform.translation.y

        # Lookup robot’s current pose in map → base_footprint (for orientation)
        try:
            robot_tf = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.base_frame,
                rclpy.time.Time())  # latest
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f"[TF] Could not get {self.base_frame} in {self.global_frame}: {ex}")
            return

        rx = robot_tf.transform.translation.x
        ry = robot_tf.transform.translation.y

        # Compute yaw so robot faces the person
        dx = px - rx
        dy = py - ry
        yaw = math.atan2(dy, dx)
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)

        # Build and publish PoseStamped goal
        goal = PoseStamped()
        goal.header.stamp    = self.get_clock().now().to_msg()
        goal.header.frame_id = self.global_frame
        goal.pose.position.x    = px
        goal.pose.position.y    = py
        goal.pose.position.z    = 0.0
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw

        self.goal_pub.publish(goal)
        self.get_logger().info(
            f"Published /goal_pose at ({px:.2f}, {py:.2f}), yaw={yaw:.2f}"
        )


def main():
    rclpy.init()
    node = PeopleFollowerGoal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
