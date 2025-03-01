import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
import serial
import time

class GripperControlService(Node):
    def __init__(self):
        super().__init__('gripper_control_service')
        # Update the serial port path according to your system
        self.ser = serial.Serial('/dev/my_gripper', 9600, timeout=1)
        # Allow time for the serial connection to establish
        time.sleep(2)
        self.srv = self.create_service(SetBool, 'gripper_control', self.gripper_callback)
        self.get_logger().info("Gripper control service is ready.")

    def gripper_callback(self, request, response):
        # If request.data is True, we want to close the gripper (send command "1").
        # If False, open the gripper (send command "0").
        command = "1\n" if request.data else "0\n"
        self.ser.write(command.encode())
        self.get_logger().info(f"Sent command: {command.strip()}")
        response.success = True
        response.message = f"Gripper command sent: {command.strip()}"
        return response

def main(args=None):
    rclpy.init(args=args)
    node = GripperControlService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
