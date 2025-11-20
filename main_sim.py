import pickle
import numpy as np
import time 

from utils.attitude import Attitude
from utils.imu_meas import IMUmeas
from utils.nav_state import NavState
from utils.vio_publisher import VIOPublisher

from msckf import MSCKF
from frontend_node import Frontend
from map_manager import MapManager

import rclpy
import matplotlib.pyplot as plt
from scipy.io import savemat

with open('sim_data.pkl', 'rb') as f:
    sim_data = pickle.load(f)
    
# Sensor config.
imu_Hz = 1/sim_data['imu']['dt']
img_Hz = 1/sim_data['img']['dt']
Nsim = len(sim_data['imu']['ts'])
Nf = sim_data['pfg']['pfg_plane'].shape[1] + sim_data['pfg']['pfg_sobj1'].shape[1] + sim_data['pfg']['pfg_sobj2'].shape[1] + sim_data['pfg']['pfg_dobj1'].shape[1] + sim_data['pfg']['pfg_dobj2'].shape[1] 

lcam = {'K' : sim_data['img']['K'], 'Tcb' : sim_data['img']['Tcb'], 'Tbc' : np.linalg.inv(sim_data['img']['Tcb']), 
        'Tlr' : sim_data['img']['Tlr'], 'Trl' : np.linalg.inv(sim_data['img']['Tlr']),
        'dc' : np.zeros(4), 'dist_model' : 'radtan'}
rcam = {'K' : sim_data['img']['K'], 'dc' : np.zeros(4), 'dist_model' : 'radtan'}

# Simulation params.
np.random.seed(10)
r_pix = 3
Rpix =r_pix**2

P0, Q = MSCKF.get_initial_uncertainty()

# Initialization params.
k_vis = 0
Nf_max = 100
px_dist = 20    
frontend = Frontend(lcam, rcam, Nf_max, px_dist)
min_slw = 3
max_slw = 10
feature_map = MapManager(min_slw, max_slw)

q0 = Attitude.euler2quat(sim_data['gt']['eul'][:, 0])
p0 = sim_data['gt']['pg'][:, 0]
v0 = sim_data['gt']['vg'][:, 0]

Ximu = NavState(q0, p0, v0, np.zeros(3), np.zeros(3), sim_data['gt']['gg'])
Xslw = []                 # list : Xslw[0]['q'] / Xslw[0]['p']
Xmsckf = MSCKF(Ximu, Xslw, min_slw, max_slw, lcam, rcam)
Xmsckf = Xmsckf.set_uncertainty(P0, Q, Rpix)

# Publisher
rclpy.init()
publisher_node = VIOPublisher()

# Noise addition
ba_true = np.linalg.cholesky(P0[9:12, 9:12]).T @ np.random.randn(3)[:,np.newaxis]     + np.cumsum(((Xmsckf.std_wa / np.sqrt(imu_Hz)) * np.random.randn(3, Nsim)), axis=1)
bg_true = np.linalg.cholesky(P0[12:15, 12:15]).T @ np.random.randn(3)[:,np.newaxis]   + np.cumsum(((Xmsckf.std_wg / np.sqrt(imu_Hz)) * np.random.randn(3, Nsim)), axis=1)

fm = sim_data['imu']['fb'] + Xmsckf.std_na * np.sqrt(imu_Hz) * np.random.randn(3, Nsim) + ba_true
wm = sim_data['imu']['wb'] + Xmsckf.std_ng * np.sqrt(imu_Hz) * np.random.randn(3, Nsim) + bg_true

# Memory allocation 
Xmsckf = Xmsckf.set_result(Nsim)

start_time = time.time()
# Main loop 
for k in range(1, Nsim):    
    ts_ = sim_data['imu']['ts'][k] 
    dt = sim_data['imu']['ts'][k] - sim_data['imu']['ts'][k-1]
    fb0_ = fm[:, k-1]   # f^b_ib
    fb1_ = fm[:, k]
    wb0_ = wm[:, k-1]   # w^b_ib
    wb1_ = wm[:, k]
    
    IMU0 = IMUmeas(fb0_, wb0_, Xmsckf.Xi.ba, Xmsckf.Xi.bg)
    IMU1 = IMUmeas(fb1_, wb1_, Xmsckf.Xi.ba, Xmsckf.Xi.bg)
    Xmsckf = Xmsckf.propagate_trapezoidal(IMU0, IMU1, dt)

    if ((k+1) % int(imu_Hz / img_Hz)) == 0:
        print(f"{k_vis} image processing ...")        
        Tgb = np.block([[Attitude.euler2dcm(sim_data['gt']['eul'][:, k]), sim_data['gt']['pg'][:, k].reshape(-1, 1)],
                        [0, 0, 0, 1]])
        Tgc = Tgb @ np.linalg.inv(frontend.lcam['Tcb'])
        
        # Point feature processing        
        frontend = frontend.set_sim_features(sim_data['gt']['pfg'], Tgc, k, k_vis)
        feature_map = feature_map.add_features(frontend.Tracks, k_vis)        
        
        publisher_node.publish_point_cloud(frontend.pf_g_true[1:].T, ts_)        
        publisher_node.publish_img(frontend.img0, frontend.img1, ts_)        
        
        if frontend.is_zupt:
            Xmsckf = Xmsckf.zupt()        
        elif feature_map.deadtracks:                 
            print(f"imu : {k}, img : {k_vis}, deadtracks : {len(feature_map.deadtracks)}")
            Xmsckf = Xmsckf.correct(feature_map.deadtracks, IMU1, 0)
            deadTracks = np.array([])
        
        publisher_node.publish_path(Xmsckf.Xi.q, Xmsckf.Xi.p, ts_) 
            
        Xmsckf = Xmsckf.augment_state(k_vis)
        k_vis = k_vis + 1
        
    Xmsckf = Xmsckf.save_result(k)
        

mat_data = {"sim_data" : sim_data, "est" : Xmsckf.est, "std" : Xmsckf.std}    
savemat("simout_wo_Fransac.mat", mat_data)

end_time = time.time()
print(f"Time elapsed : {end_time - start_time:.6f} [sec]")


