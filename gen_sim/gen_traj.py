import sys
sys.path.append('/home/nesl/sim_vio')

import numpy as np 

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial.transform import Rotation as R

from utils.attitude import Attitude
from gen_features import GenFeatures

import pickle

# Initial variables
gg = np.array([0, 0, 9.8])
imu_hz = 200                    # [hz]
img_hz = 20                     # [hz]
dt = 1/imu_hz                   # [sec]
ts = 60                         # [sec]
t = np.arange(0, ts + dt, dt)       # 0:dt:ts
N = len(t)

# Trajectory
s_traj = 20 * np.pi * 2
v_const = s_traj/ts

# Attitude 
eul = np.zeros((3, N))                      # zeros input should be tuple
eul[2, :] = np.linspace(0, -2 * np.pi, N)   # NED & yaw changes

# Angular rate
wb = np.zeros((3, N))
del_Rgb0 = np.zeros((3, 3, N))
del_Rgb1 = np.zeros((3, 3, N))
for i in range(1, N):
    Rgb0 = Attitude.euler2dcm(eul[:, i-1])      # eul^g_b = R^g_b
    Rgb1 = Attitude.euler2dcm(eul[:, i])
    
    w_X = Rgb0.T @ (Rgb1 - Rgb0) / dt             # from Rgb1 = Rgb0 * exp(I+w_X*dt) = Rgb0 * (I + w_X*dt)
    wb[:, i] = np.array([w_X[2,1], w_X[0,2], w_X[1,0]])     # w1, w2, w3

# Velocity
vg = np.zeros((3, N))
vb = np.zeros((3, N))
vb[1, :] = v_const * np.ones((1, N))                        # only y-axis accel.
for i in range(N):                                          # rotate every vb to vg
    Rgb_ = Attitude.euler2dcm(eul[:, i])
    vg[:, i] = Rgb_ @ vb[:, i]    

# Specific forces
fb = np.zeros((3, N))
for i in range(1, N):
    ag_ = (vg[:, i] - vg[:, i-1]) / dt
    Rgb_ = Attitude.euler2dcm(eul[:, i-1])    
    fb[:, i-1] = Rgb_.T @ (ag_ - gg)
fb[:, -1] = fb[:, -2]

# Position
pg = np.zeros((3, N))
pg[:, 0] = [-20, 0, 0]
for i in range(1, N):
    pg[:, i] = pg[:, i-1] + vg[:, i-1] * dt

# Global Point features of objects : (center_x, center_y, center_z, length, width, height, num_points)
sobj1_args = np.array([10, -10, 0, 6, 4, 2, 50])
sobj2_args = np.array([0, 10, 0, 4, 6, 4, 50])
dobj1_args = np.array([15, 0, 0, 4, 5, 3, 50])
dobj2_args = np.array([-5, -15, 0, 5, 5, 3, 50])
dobj3_args = np.array([-15, 0, 0, 3, 3, 5, 50])

pfg_plane = GenFeatures.generate_plane_features()

pfg_sobj1 = GenFeatures.generate_random_points_on_box(sobj1_args, pfg_plane.shape[1], 20)
pfg_sobj2 = GenFeatures.generate_random_points_on_box(sobj2_args, pfg_plane.shape[1] + pfg_sobj1.shape[1], 21)
pfg_dobj1 = GenFeatures.generate_random_points_on_box(dobj1_args, pfg_plane.shape[1] + pfg_sobj1.shape[1] + pfg_sobj2.shape[1], 30)
pfg_dobj2 = GenFeatures.generate_random_points_on_box(dobj2_args, pfg_plane.shape[1] + pfg_sobj1.shape[1] + pfg_sobj2.shape[1] + pfg_dobj1.shape[1], 31)
pfg_dobj3 = GenFeatures.generate_random_points_on_box(dobj3_args, pfg_plane.shape[1] + pfg_sobj1.shape[1] + pfg_sobj2.shape[1] + pfg_dobj1.shape[1] + pfg_dobj2.shape[1], 32)

v_dobj1 = -0.5 * np.array([np.sqrt(3)/2, 1/2, 0])                        # [m/sec]
v_dobj2 =  0.5 * np.array([-0, 1, 0])                       # [m/sec]
v_dobj3 =  0.5 * np.array([np.sqrt(3)/2, 1/2, 0])                     # [m/sec]

pfg_dobj1_mat = np.zeros((pfg_dobj1.shape[0], pfg_dobj1.shape[1], N))
pfg_dobj2_mat = np.zeros((pfg_dobj2.shape[0], pfg_dobj2.shape[1], N))
pfg_dobj3_mat = np.zeros((pfg_dobj3.shape[0], pfg_dobj3.shape[1], N))
pfg_dobj1_mat[:, :, 0] = pfg_dobj1
pfg_dobj2_mat[:, :, 0] = pfg_dobj2
pfg_dobj3_mat[:, :, 0] = pfg_dobj3
for i in range(1, N):
    pfg_dobj1_mat[0, :, i] = pfg_dobj1_mat[0, :, 0]
    pfg_dobj1_mat[4, :, i] = pfg_dobj1_mat[4, :, 0]
    pfg_dobj1_mat[1:4, :, i] = pfg_dobj1_mat[1:4, :, i-1] + v_dobj1[:, np.newaxis] * dt     # broadcasting     
    
    pfg_dobj2_mat[0, :, i] = pfg_dobj2_mat[0, :, 0]
    pfg_dobj2_mat[4, :, i] = pfg_dobj2_mat[4, :, 0]
    pfg_dobj2_mat[1:4, :, i] = pfg_dobj2_mat[1:4, :, i-1] + v_dobj2[:, np.newaxis] * dt     # broadcasting 
    
    pfg_dobj3_mat[0, :, i] = pfg_dobj3_mat[0, :, 0]
    pfg_dobj3_mat[4, :, i] = pfg_dobj3_mat[4, :, 0]
    pfg_dobj3_mat[1:4, :, i] = pfg_dobj3_mat[1:4, :, i-1] + v_dobj3[:, np.newaxis] * dt     # broadcasting 

# Camera config.
resl = np.array([752, 480])
K = np.array([[360,  0,      resl[0]/2],
             [0,    360,    resl[1]/2],
             [0,    0,      1]])
Tcb = np.array([[0, 1,  0,  -0.05],
                [0, 0,  1,  0.05],
                [1, 0,  0,  -0.03],
                [0, 0,  0,  1]])
Tlr = np.array([[1, 0,  0,  0.2],
                [0, 1,  0,  0],
                [0, 0,  1,  0],
                [0, 0,  0,  1]])
dt_img = 1/20   # 20 hz

# Assignment
pfg = {'pfg_plane' : pfg_plane, 'pfg_sobj1' : pfg_sobj1, 'pfg_sobj2' : pfg_sobj2, 'pfg_dobj1' : pfg_dobj1_mat, 'pfg_dobj2' : pfg_dobj2_mat}
gt = {'pg' : pg, 'eul' : eul, 'vg' : vg, 'vb' : vb, 'pfg' : pfg, 'gg' : gg}
imu = {'ts' : t, 'gg' : gg, 'wb' : wb, 'fb' : fb, 'dt' : dt}
img = {'resl' : resl, 'K' : K, 'Tcb' : Tcb, 'Tlr' : Tlr, 'dt' : dt_img}
sim_data = {'pfg' : pfg, 'gt' : gt, 'imu' : imu, 'img' : img}

# with open('sim_data.pkl', 'wb') as f:
#     pickle.dump(sim_data, f)
    
# with open('sim_data.pkl', 'rb') as f:
#     loaded_data = pickle.load(f)
    
# Simulation Environment 
fig0 = plt.figure()
ax = fig0.add_subplot(111, projection='3d')
ax.plot3D(pg[0, :], pg[1, :], pg[2, :], 'k-')
ax.scatter(pfg_plane[1, :], pfg_plane[2, :], pfg_plane[3, :], color='b', marker='.', s=10, label="Plane")
ax.scatter(pfg_sobj1[1, :], pfg_sobj1[2, :], pfg_sobj1[3, :], color='g', marker='.', s=10, label="Static Object 1")
ax.scatter(pfg_sobj2[1, :], pfg_sobj2[2, :], pfg_sobj2[3, :], color='g', marker='.', s=10, label="Static Object 2")
ax.set_xlabel('x^g [m]'); ax.set_ylabel('y^g [m]'); ax.set_zlabel('z^g [m]'); 
ax.grid(True)
ax.set_zlim([-5, 5])
ax.set_box_aspect([1, 1, 0.2])

pg_dot = ax.scatter(pg[0, 0], pg[1, 0], pg[2, 0], color='k', marker='o', label="Current position")
scatter_dobj1 = ax.scatter(pfg_dobj1_mat[1, :, 0], pfg_dobj1_mat[2, :, 0], pfg_dobj1_mat[3, :, 0], color='r', marker='.', s=10, label="Dynamic Object 1")  
scatter_dobj2 = ax.scatter(pfg_dobj2_mat[1, :, 0], pfg_dobj2_mat[2, :, 0], pfg_dobj2_mat[3, :, 0], color='r', marker='.', s=10, label="Dynamic Object 2")   
scatter_dobj3 = ax.scatter(pfg_dobj3_mat[1, :, 0], pfg_dobj3_mat[2, :, 0], pfg_dobj3_mat[3, :, 0], color='r', marker='.', s=10, label="Dynamic Object 3")   

def init():
    pg_dot._offsets3d = ([], [], [])
    scatter_dobj1._offsets3d = ([], [], [])
    scatter_dobj2._offsets3d = ([], [], [])
    scatter_dobj3._offsets3d = ([], [], [])
    return scatter_dobj1, scatter_dobj2, scatter_dobj3

def update(frame):
    pg_dot._offsets3d = ([pg[0, imu_hz * frame]], [pg[1, imu_hz * frame]], [pg[2, imu_hz * frame]])
    scatter_dobj1._offsets3d = (pfg_dobj1_mat[1, :, imu_hz * frame], pfg_dobj1_mat[2, :, imu_hz * frame], pfg_dobj1_mat[3, :, imu_hz * frame])
    scatter_dobj2._offsets3d = (pfg_dobj2_mat[1, :, imu_hz * frame], pfg_dobj2_mat[2, :, imu_hz * frame], pfg_dobj2_mat[3, :, imu_hz * frame])
    scatter_dobj3._offsets3d = (pfg_dobj3_mat[1, :, imu_hz * frame], pfg_dobj3_mat[2, :, imu_hz * frame], pfg_dobj3_mat[3, :, imu_hz * frame])
    return scatter_dobj1, scatter_dobj2, scatter_dobj3

ani = FuncAnimation(fig0, func=update, frames=range(0, 60), init_func=init, interval=0, blit=False)
plt.show()