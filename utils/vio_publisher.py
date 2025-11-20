# import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
import sensor_msgs_py.point_cloud2 as pcl2
from std_msgs.msg import Header
from rclpy.time import Time

import struct
import cv2
from cv_bridge import CvBridge

import numpy as np

class VIOPublisher(Node):
    def __init__(self):
        super().__init__('vio_publisher')
        
        self.odom_pub = self.create_publisher(Odometry, 'odom_msckf', 10)
        self.path_pub = self.create_publisher(Path, 'path_msckf', 10)
        self.img0_pub = self.create_publisher(Image, 'img0', 10)
        self.img1_pub = self.create_publisher(Image, 'img1', 10)
        self.pcl_pub = self.create_publisher(PointCloud2, 'global_feature_point', 10)
        
        self.bridge = CvBridge()   
        self.path_msg = Path()   
        self.path_msg.header.frame_id = 'global'  
    
    def publish_odom(self, q_, p_, ts_):
        seconds_ = int(ts_)
        nanoseconds_ = int((ts_ - seconds_) * 1e9)
        ts = Time(seconds=seconds_, nanoseconds=nanoseconds_)
                
        odom_msg = Odometry()
        # odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.stamp = ts.to_msg()
        odom_msg.header.frame_id = 'global'
        
        odom_msg.pose.pose.position.x = p_[0]
        odom_msg.pose.pose.position.y = p_[1]
        odom_msg.pose.pose.position.z = p_[2]
        
        odom_msg.pose.pose.orientation.w = q_[0]
        odom_msg.pose.pose.orientation.x = q_[1]
        odom_msg.pose.pose.orientation.y = q_[2]
        odom_msg.pose.pose.orientation.z = q_[3]
        
        self.odom_pub.publish(odom_msg)        
        self.get_logger().info('MSCKF Odometry Published ...')
        
    def publish_img(self, img0_, img1_, ts_):
        """ Stereo image publisher
        Args:
            img0_ (cv2.Mat): left image
            img1_ (cv2.Mat): right image
        """
        seconds_ = int(ts_)
        nanoseconds_ = int((ts_ - seconds_) * 1e9)
        ts = Time(seconds=seconds_, nanoseconds=nanoseconds_)
        
        img0_msg_ = self.bridge.cv2_to_imgmsg(img0_, encoding="bgr8")
        img1_msg_ = self.bridge.cv2_to_imgmsg(img1_, encoding="bgr8")        
        img0_msg_.header.stamp = img1_msg_.header.stamp = ts.to_msg()
        self.img0_pub.publish(img0_msg_)
        self.img1_pub.publish(img1_msg_)        
        # self.get_logger().info('Stereo Images Published ...')
        
    def publish_path(self, q_, p_, ts_):
        seconds_ = int(ts_)
        nanoseconds_ = int((ts_ - seconds_) * 1e9)
        ts = Time(seconds=seconds_, nanoseconds=nanoseconds_)
        
        pose = PoseStamped()
        pose.header.stamp = ts.to_msg()
        pose.header.frame_id = 'global'
        
        pose.pose.position.x = p_[0].real
        pose.pose.position.y = p_[1].real
        pose.pose.position.z = p_[2].real
        
        pose.pose.orientation.w = q_[0].real
        pose.pose.orientation.x = q_[1].real
        pose.pose.orientation.y = q_[2].real
        pose.pose.orientation.z = q_[3].real
        
        self.path_msg.poses.append(pose)
        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(self.path_msg)
        # self.get_logger().info('MSCKF Path Published ...')
        
    def publish_point_cloud(self, points, ts_):
        seconds_ = int(ts_)
        nanoseconds_ = int((ts_ - seconds_) * 1e9)
        ts = Time(seconds=seconds_, nanoseconds=nanoseconds_)        
        
        header = Header()
        header.stamp = ts.to_msg()
        header.frame_id = 'global'        
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        if points.shape[1] == 4:  # If intensity is provided
            fields.append(PointField(name='intensity', offset=12, count=1, datatype=PointField.FLOAT32))
        
        pcl_data = pcl2.create_cloud(header=header, fields=fields, points=points)
        self.pcl_pub.publish(pcl_data)
        # self.get_logger().info('Global feature points Published ...')
        