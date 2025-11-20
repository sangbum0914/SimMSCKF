import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from cv_bridge import CvBridge



class RosbagReader:
    reader = []
    cv_bridge = []
    imu_msg_type = []
    img_msg_type = []
    
    imu_topic   = ""
    img0_topic  = ""
    img1_topic  = ""
    simg0_topic = ""
    
    def __init__(self, bag_path_, configs_):        
        storage_options = rosbag2_py.StorageOptions(uri=bag_path_, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        
        self.reader = rosbag2_py.SequentialReader()
        self.reader.open(storage_options, converter_options)
        
        topic_type_map = {topic.name: topic.type for topic in self.reader.get_all_topics_and_types()}
        
        self.imu_msg_type = get_message(topic_type_map[configs_['imu_topic']])
        self.img_msg_type = get_message(topic_type_map[configs_['image0_topic']])
        self.gt_msg_type = get_message(topic_type_map['/odometry'])
        
        self.imu_topic = configs_['imu_topic']
        self.img0_topic = configs_['image0_topic']
        self.img1_topic = configs_['image1_topic']
        self.simg0_topic = configs_['seg_image0_topic']
        self.gt_topic = '/odometry'
        self.cv_bridge = CvBridge()        
    
    def process_bag(self, topic_, data_, timestamp_):
        imu = img0 = img1 = simg0 = gt = []
        
        if topic_ == self.imu_topic :
            imu_msg = deserialize_message(data_, self.imu_msg_type)
            ts = imu_msg.header.stamp.sec + imu_msg.header.stamp.nanosec*1e-9
            fb = np.array([imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z])
            wb = np.array([imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z])
            imu = {'ts': ts, 'fb': fb, 'wb': wb}
            
        elif topic_ == self.img0_topic:                    
            img0_msg = deserialize_message(data_, self.img_msg_type)
            ts = img0_msg.header.stamp.sec + img0_msg.header.stamp.nanosec*1e-9
            img0_ = self.cv_bridge.imgmsg_to_cv2(img0_msg, desired_encoding='rgb8')
            img0 = {'ts': ts, 'img': img0_}
            
        elif topic_ == self.img1_topic:        
            img1_msg = deserialize_message(data_, self.img_msg_type)
            ts = img1_msg.header.stamp.sec + img1_msg.header.stamp.nanosec*1e-9
            img1_ = self.cv_bridge.imgmsg_to_cv2(img1_msg, desired_encoding='rgb8')
            img1 = {'ts': ts, 'img': img1_}
            
        elif topic_ == self.simg0_topic:        
            simg0_msg = deserialize_message(data_, self.img_msg_type)
            ts = simg0_msg.header.stamp.sec + simg0_msg.header.stamp.nanosec*1e-9
            simg0_ = self.cv_bridge.imgmsg_to_cv2(simg0_msg, desired_encoding='rgb8')
            simg0 = {'ts': ts, 'img': simg0_}     
        
        elif topic_ == self.gt_topic:
            gt_msg = deserialize_message(data_, self.gt_msg_type)
            ts = gt_msg.header.stamp.sec + gt_msg.header.stamp.nanosec*1e-9
            q = np.array([gt_msg.pose.pose.orientation.w, gt_msg.pose.pose.orientation.x, gt_msg.pose.pose.orientation.y, gt_msg.pose.pose.orientation.z])
            p = np.array([gt_msg.pose.pose.position.x, gt_msg.pose.pose.position.y, gt_msg.pose.pose.position.z])           
            gt = {'ts': ts, 'q': q, 'p': p}
            
        return imu, img0, img1, simg0, gt 