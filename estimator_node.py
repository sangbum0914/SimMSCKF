import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json 

class estimator_node(Node):
    def __init__(self):
        super().__init__('estimator_node') 
        self.sfeas_sub = self.create_subscription(String, 'feas_sub', self.sfeas_callback, 10)
        
    def sfeas_callback(self, msg):
        sfeas_list = json.loads(msg.data)
        self.get_logger().info(f"Received {sfeas_list}")
        
def main(args=None):
    rclpy.init(args=args)
    node = estimator_node()
    try:
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down subscriber.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
