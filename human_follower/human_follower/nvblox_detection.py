#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import cv2
import json

# Compute human centroid using the colored UNet mask and depth
class MaskCentroidNode(Node):
    def __init__(self):
        super().__init__('mask_centroid_node')
        self.bridge = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None
        self.latest_depth = None
        qos = QoSPresetProfiles.SENSOR_DATA.value

        # Subscribe to full-res camera info
        self.get_logger().info('Subscribing to /camera0/depth/camera_info...')
        self.create_subscription(
            CameraInfo,
            '/camera0/depth/camera_info',
            self.caminfo_cb,
            qos
        )
        # Subscribe to full-res depth
        self.get_logger().info('Subscribing to /camera0/depth/image_rect_raw...')
        self.create_subscription(
            Image,
            '/camera0/depth/image_rect_raw',
            self.depth_cb,
            qos
        )
        # Subscribe to colored segmentation mask
        self.get_logger().info('Subscribing to /camera0/unet/colored_segmentation_mask...')
        self.create_subscription(
            Image,
            '/camera0/unet/colored_segmentation_mask',
            self.mask_cb,
            qos
        )

        # Publisher for centroid as JSON string array
        self.pub = self.create_publisher(String,
                                         '/human_centroid_str',
                                         10)
        self.get_logger().info('Publisher /human_centroid_str ready')

    def caminfo_cb(self, msg: CameraInfo):
        # Load intrinsics once
        if self.fx is None:
            self.fx, self.fy = msg.k[0], msg.k[4]
            self.cx, self.cy = msg.k[2], msg.k[5]
            self.get_logger().info(
                f'Intrinsics loaded: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}'
            )

    def depth_cb(self, msg: Image):
        # Store latest full-res depth in meters
        depth_img = self.bridge.imgmsg_to_cv2(msg, 'passthrough').astype(np.float32)
        # Some cameras output in mm; detect and convert if needed
        if np.max(depth_img) > 1000:
            depth_img *= 0.001
        self.latest_depth = depth_img

    def mask_cb(self, msg: Image):
        # Ensure both depth and intrinsics are ready
        if self.latest_depth is None or self.fx is None:
            self.get_logger().warn('Waiting for both depth and intrinsics...')
            return

        # Convert colored mask to binary
        rgb = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask_small = gray > 0

        # Resize mask to depth resolution
        h_dep, w_dep = self.latest_depth.shape
        mask = cv2.resize(
            mask_small.astype(np.uint8),
            (w_dep, h_dep),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        # Filter out invalid depths
        valid_depth = (self.latest_depth > 0.2) & (self.latest_depth < 5.0)
        mask &= valid_depth

        ys, xs = np.where(mask)
        if ys.size == 0:
            self.get_logger().info('No valid human pixels detected')
            return

        # Back-project to camera frame
        zs = self.latest_depth[ys, xs]
        xs_cam = (xs - self.cx) * zs / self.fx
        ys_cam = (ys - self.cy) * zs / self.fy
        pts = np.vstack((xs_cam, ys_cam, zs)).T

        # Use median for robustness
        centroid = np.median(pts, axis=0)
        x_c, y_c, z_c = centroid.tolist()

        # Log detailed stats
        self.get_logger().info(
            f'Centroid (m): x={x_c:.3f}, y={y_c:.3f}, z={z_c:.3f} '
            f'from {pts.shape[0]} points'
        )

        # Publish as JSON string array
        out = String()
        out.data = json.dumps([f'{x_c:.3f}', f'{y_c:.3f}', f'{z_c:.3f}'])
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = MaskCentroidNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
