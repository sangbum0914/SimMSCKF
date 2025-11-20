from utils.attitude import Attitude

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
        self.feas_img0_pub = self.create_publisher(Image, 'feas_img0', 10)
        self.seg_img0_pub = self.create_publisher(Image, 'seg_img0', 10)
        self.pcl_pub = self.create_publisher(PointCloud2, 'global_feature_point', 10)
        self.gt_pub = self.create_publisher(Path, 'path_gt', 10)
        
        self.bridge = CvBridge()   
        self.path_msg_est = Path()   
        self.path_msg_gt = Path()   
        self.path_msg_est.header.frame_id = self.path_msg_gt.header.frame_id = 'global'          
        
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
        
    def publish_img(self, feas_img0_, seg_img0_, ts_):
        """ features & segmented image publisher
        Args:
            img0_ (cv2.Mat): left image
            img1_ (cv2.Mat): right image
        """
        seconds_ = int(ts_)
        nanoseconds_ = int((ts_ - seconds_) * 1e9)
        ts = Time(seconds=seconds_, nanoseconds=nanoseconds_)
        
        feas_img0_msg_ = self.bridge.cv2_to_imgmsg(feas_img0_, encoding="bgr8")
        seg_img0_msg_ = self.bridge.cv2_to_imgmsg(seg_img0_, encoding="bgr8")        
        feas_img0_msg_.header.stamp = seg_img0_msg_.header.stamp = ts.to_msg()
        self.feas_img0_pub.publish(feas_img0_msg_)
        self.seg_img0_pub.publish(seg_img0_msg_)        
        # self.get_logger().info('Stereo Images Published ...')
        
    def publish_path(self, q_, p_, ts_):
        """Publish path in ENU frame
        Args:
            q_ (_type_): _description_
            p_ (_type_): _description_
            ts_ (_type_): _description_
        """
        seconds_ = int(ts_)
        nanoseconds_ = int((ts_ - seconds_) * 1e9)
        ts = Time(seconds=seconds_, nanoseconds=nanoseconds_)
        
        pose = PoseStamped()
        pose.header.stamp = ts.to_msg()
        pose.header.frame_id = 'global'
        
        # Change the global frame (NED) to ENU fame
        # R_ENU_NED = np.array([[0, 1, 0],
        #                       [1, 0, 0],
        #                       [0, 0, -1]])      # Convert NED frame to ENU frame
        # q_ = Attitude.dcm2quat(R_ENU_NED @ Attitude.quat2dcm(q_))
        # p_ = R_ENU_NED @ p_
        
        pose.pose.position.x = p_[0].real
        pose.pose.position.y = p_[1].real
        pose.pose.position.z = p_[2].real
        
        pose.pose.orientation.w = q_[0].real
        pose.pose.orientation.x = q_[1].real
        pose.pose.orientation.y = q_[2].real
        pose.pose.orientation.z = q_[3].real
        
        self.path_msg_est.poses.append(pose)
        self.path_msg_est.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(self.path_msg_est)
        # self.get_logger().info('MSCKF Path Published ...')
        
    def publish_gt(self, q_, p_,ts_):
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
        
        self.path_msg_gt.poses.append(pose)
        self.path_msg_gt.header.stamp = self.get_clock().now().to_msg()
        self.gt_pub.publish(self.path_msg_gt)
        
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
        