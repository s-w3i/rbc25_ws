#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, TransformStamped
from robot_interfaces.srv import ComputeGoalPose
import tf2_ros
from tf_transformations import quaternion_from_euler


class ComputeGoalPoseNode(Node):
    def __init__(self):
        super().__init__('compute_goal_pose_node')

        # frames
        self.global_frame = 'map'
        self.base_frame = 'base_footprint'

        # desired standoff distance in meters
        self.standoff_distance = 1.5

        # TF buffer / listener (for robot pose)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # cache of latest transforms by child_frame_id
        self.transforms = {}

        # subscribe both people and chair transforms into the same cache
        self.create_subscription(
            TransformStamped,
            '/people_transform',
            self._tf_callback,
            10
        )
        self.create_subscription(
            TransformStamped,
            '/chair_transform',
            self._tf_callback,
            10
        )

        # publisher for one-shot goals
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # service to compute & publish a goal for a given target_frame
        self.create_service(
            ComputeGoalPose,
            '/compute_goal_pose',
            self._on_compute_goal
        )

        self.get_logger().info('ComputeGoalPoseNode ready.')

    def _tf_callback(self, msg: TransformStamped):
        # store or overwrite the latest transform for this frame
        self.transforms[msg.child_frame_id] = msg

    def _on_compute_goal(self, request, response):
        target = request.target_frame

        # check we have a transform for this frame
        if target not in self.transforms:
            response.success = False
            response.message = f"No TransformStamped received for '{target}'"
            return response

        # extract raw target position in map frame
        tf_msg = self.transforms[target]
        tx = tf_msg.transform.translation.x
        ty = tf_msg.transform.translation.y

        # look up robot's current pose (map -> base_footprint)
        try:
            robot_tf = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.base_frame,
                rclpy.time.Time()
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            response.success = False
            response.message = f"TF lookup failed: {e}"
            return response

        rx = robot_tf.transform.translation.x
        ry = robot_tf.transform.translation.y

        # compute vector & yaw so robot faces the target
        dx = tx - rx
        dy = ty - ry
        dist = math.hypot(dx, dy)
        yaw = math.atan2(dy, dx)
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)

        # truncate so we stop standoff_distance from the target
        if dist > self.standoff_distance:
            travel = dist - self.standoff_distance
            gx = rx + math.cos(yaw) * travel
            gy = ry + math.sin(yaw) * travel
        else:
            gx, gy = rx, ry
            self.get_logger().warn(
                f"Target '{target}' is only {dist:.2f} m away (< {self.standoff_distance} m); "
                "publishing current pose as goal."
            )

        # build and publish the truncated goal
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = self.global_frame
        goal.pose.position.x = gx
        goal.pose.position.y = gy
        goal.pose.position.z = 0.0
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw

        self.goal_pub.publish(goal)

        # fill in the service response
        response.goal_pose = goal
        response.success = True
        response.message = (
            f"Published goal for '{target}' at ({gx:.2f}, {gy:.2f}), "
            f"{self.standoff_distance} m from actual pose."
        )
        return response


def main():
    rclpy.init()
    node = ComputeGoalPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
