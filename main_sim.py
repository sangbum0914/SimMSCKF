import numpy as np
import time 
import os

from utils.attitude import Attitude
from utils.imu_meas import IMUmeas
from utils.nav_state import NavState
from utils.vio_publisher import VIOPublisher
from utils.rosbag_reader import RosbagReader
from utils.yaml_reader import YAMLReader

from msckf import MSCKF
from frontend import Frontend
from static_map import StaticMap
from object_processor import ObjectProcessor

import rclpy
import matplotlib.pyplot as plt
from scipy.io import savemat

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import cv2

# YAML config.
dir = os.path.dirname(os.path.abspath(__file__))
config_file = "VIODE.yaml"
configs = YAMLReader(dir, config_file)

# constructors
rclpy.init()
publisher_node = VIOPublisher()
Ximu = NavState(np.zeros(4), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.array([0, 0, configs['g_norm']]))
Xslw = []                 # list : Xslw[0]['q'] / Xslw[0]['p']
Xmsckf = MSCKF(Ximu, Xslw, configs)
frontend = Frontend(configs)
static_map = StaticMap(configs['min_slw'], configs['max_slw'])
object_processor = ObjectProcessor(configs['min_slw'], configs['max_slw'], configs['lcam'], configs['rcam'])

# Ros2 bag file read
bag_path = '/media/nesl/D_drive/VIODE/bag2/parking_lot/3_high/'
rosbag = RosbagReader(bag_path_=bag_path, configs_=configs)

imgs = {'img0': {}, 'img1': {}, 'simg0': {}}
k_vis = 0
start_time = time.time()
while rosbag.reader.has_next():
    (topic, data, timestamp) = rosbag.reader.read_next()
    imu, img0, img1, simg0, gt = rosbag.process_bag(topic, data, timestamp)
        
    if not Xmsckf.is_aligned:        
        if imu or img0:
            Xmsckf = Xmsckf.initial_alignment(imu, img0) # INS initial alignment
    else:                      
        if imu:
            IMU = IMUmeas(imu['ts'], imu['fb'], imu['wb'], Xmsckf.Xi.ba, Xmsckf.Xi.bg)
            Xmsckf = Xmsckf.propagate_trapezoidal(IMU, imu['ts'])
            Xmsckf = Xmsckf.save_result(imu['ts'])
        elif img0:
            imgs['img0'] = img0
        elif img1:
            imgs['img1'] = img1
        elif simg0:
            imgs['simg0'] = simg0
        elif gt:
            publisher_node.publish_gt(q_=gt['q'], p_=gt['p'],ts_=gt['ts'])
                    
        if all(bool(value) for value in imgs.values()):
            if (np.abs(imgs['img1']['ts'] - imgs['img0']['ts']) < 0.01) and (np.abs(imgs['simg0']['ts'] - imgs['img0']['ts']) < 0.01):
                frontend = frontend.set_images(imgs['img0']['img'], imgs['img1']['img'], imgs['simg0']['img'], imgs['img0']['ts'], k_vis)
                static_map = static_map.add_features(frontend.sTracks, k_vis)
                object_processor = object_processor.detect_dynamic_and_mapping(frontend.oTracks, k_vis, Xmsckf, imgs['img0']['ts'])
                
                if frontend.is_zupt:
                    Xmsckf = Xmsckf.zupt()        
                elif static_map.dead_tracks or object_processor.dead_objects:                
                    publisher_node.get_logger().info(f'Measurement correction with {len(static_map.dead_tracks)} deadtracks')
                    Xmsckf = Xmsckf.correct(static_map.dead_tracks, object_processor.dead_objects, IMU, imgs['img0']['ts'])
                
                # Xmsckf = Xmsckf.save_result(imgs['img0']['ts'])
                Xmsckf = Xmsckf.augment_state(k_vis)
                k_vis += 1
                
                publisher_node.publish_path(Xmsckf.Xi.q, Xmsckf.Xi.p, imgs['img0']['ts']) 
                publisher_node.publish_img(feas_img0_=frontend.fimg0, seg_img0_=imgs['simg0']['img'], ts_=imgs['img0']['ts'])
                imgs = {'img0': {}, 'img1': {}, 'simg0': {}}                
            else:
                ValueError('Time synchronized is needed for each images')
            
        
            
