import numpy as np

from utils.attitude import Attitude
from utils.imu_meas import IMUmeas
from utils.nav_state import NavState
from utils.SO3 import SO3
from frontend import Frontend

from scipy.stats import chi2
from scipy.optimize import least_squares
import scipy.linalg
import time

class MSCKF:
    chi_trust = 0.95
    Dxi = 15
    Dxs = 6
    Xi_idx = slice(0, 15)
    q_idx = slice(0, 3)
    p_idx = slice(3, 6)
    v_idx = slice(6, 9)
    ba_idx = slice(9, 12)
    bg_idx = slice(12, 15)
    sq_idx = slice(0, 3)
    sp_idx = slice(3, 6)
    
    d2r = np.pi/180    
    """
    # Noise specification of ADIS16448 from (https://www.analog.com/media/en/technical-documentation/data-sheets/adis16448.pdf)
    std_na = 0.20e-3/9.81;                      # [m/s^2]     : mg              : Accel. bias repetability (White noise)    : 20 mg       
    std_ng = 0.5*d2r/3600;                      # [rad/sec]   : deg/hr          : Gyro.  bias repetability (White noise)    : 0.5 deg/hr
    std_wa = 0.11e-3*9.81/np.sqrt(100);         # [m/s^4]     : mg/sqrt(Hz)     : Vel. random walk                          : 0.11 mg/sqrt(Hz)  
    std_wg = (14.5*d2r)/3600/np.sqrt(100);      # [rad/sec^3] : deg/s/sqrt(Hz)  : Ang. random walk                          : 14.5 deg/hr/sqrt(Hz)
    """
    
    def __init__(self, Ximu_, Xslw_, configs_):
        self.Xi = Ximu_
        self.Xs = Xslw_             # list 
        self.min_slw = configs_['min_slw']
        self.max_slw = configs_['max_slw']
        self.lcam = configs_['lcam']
        self.rcam = configs_['rcam']       
        self.static_threshold = configs_['static_threshold']
        self.F_threshold = configs_['F_threshold']
        
        # Covariances
        self.P = np.diag(np.hstack([configs_['P0/q']**2*np.ones(3), configs_['P0/p']**2*np.ones(3), configs_['P0/v']**2*np.ones(3), 
                configs_['P0/ba']**2*np.ones(3), configs_['P0/bg']**2*np.ones(3)]))
        self.Q = np.diag(np.hstack([configs_['Q/ng']**2*np.ones(3), np.zeros(3), configs_['Q/na']**2*np.ones(3), 
                configs_['Q/wa']**2*np.ones(3), configs_['Q/wg']**2*np.ones(3)]))               
        self.Rpix = configs_['R/nf']
        self.Rs = configs_['R/nf']**2 * np.eye(4) 
        self.Rd = (3*configs_['R/nf'])**2 * np.eye(4)   
        
        # Initial paramse
        self.is_init = True
        self.is_aligned = False
        self.is_vehicle_static_past = False
        self.imu0 = {'ts': [], 'fb': [], 'wb': []}        
        self.pIMU = IMUmeas(0, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))        
        self.est = {'ts': [], 'q': [], 'p': [], 'v': [], 'ba': [], 'bg': []}
        self.std = {'q': [], 'p': [], 'v': [], 'ba': [], 'bg': []}
               
    def initial_alignment(self, imu_, img0_):        
        if self.is_init and np.any(img0_):            
            self.pimg = img0_
            self.is_init = False                
        elif not self.is_init:
            if imu_:
                self.imu0['ts'].append(imu_['ts'])
                self.imu0['fb'].append(imu_['fb'])
                self.imu0['wb'].append(imu_['wb'])
                
            elif img0_:
                is_vehicle_static = Frontend.check_static(self.pimg['img'], img0_['img'], self.lcam, self.rcam, self.F_threshold, self.static_threshold)
                self.pimg = img0_
                
                if (self.is_vehicle_static_past and is_vehicle_static) and ((self.imu0['ts'][-1] - self.imu0['ts'][0]) > 0.25): 
                        fb0 = np.array(self.imu0['fb'])
                        wb0 = np.array(self.imu0['wb'])
                        q0, bg0 = Attitude.align_INS(fb0, wb0)
                        print(f"INS initial alignment with {len(self.imu0['ts'])} imu samples...")
                        # publisher_node.get_logger().info('Successful initial alignment of INS') 
                        self.Xi.q = q0
                        self.Xi.bg = bg0
                        self.is_aligned = True
                        self.is_init    = True # for filter propagation
                        self.p_ts = img0_['ts']
                        del self.imu0                
                
                elif self.is_vehicle_static_past and not is_vehicle_static:
                    if (self.imu0['ts'][-1] - self.imu0['ts'][0]) < 0.5:        # At least, 50 imu samples are needed.
                        ValueError("Vehicle should be temporarily static for initial alignment...")
                
                elif not self.is_vehicle_static_past and not is_vehicle_static:
                    self.imu0['ts'] = []
                    self.imu0['fb'] = []
                    self.imu0['wb'] = []        
                
                self.is_vehicle_static_past = is_vehicle_static                  
        return self
           
    def save_result(self, ts_):
        self.est['ts'].append(ts_)
        self.est['q'].append(self.Xi.q)
        self.est['p'].append(self.Xi.p)
        self.est['v'].append(self.Xi.v)
        self.est['ba'].append(np.zeros(3))
        self.est['bg'].append(np.zeros(3))
        
        self.std['q'].append(np.sqrt(np.diag(self.P[0:3, 0:3])))
        self.std['p'].append(np.sqrt(np.diag(self.P[3:6, 3:6])))        
        self.std['v'].append(np.sqrt(np.diag(self.P[6:9, 6:9])))        
        self.std['ba'].append(np.sqrt(np.diag(self.P[9:12, 9:12])))        
        self.std['bg'].append(np.sqrt(np.diag(self.P[12:15, 12:15])))        
        return self
    
    def propagate(self, IMU0_, ts_):
        Xs_idx = range(MSCKF.Dxi, self.P.shape[0])
        
        Pi = self.P[MSCKF.Xi_idx, MSCKF.Xi_idx]
        self.Xi, Pi, F = MSCKF.propagate_imu_euler(self.Xi, Pi, IMU0_, self.Q, dt_=ts_ - self.p_ts)
        self.P[MSCKF.Xi_idx, MSCKF.Xi_idx] = Pi
        
        if self.P.shape[0] > MSCKF.Dxi:
            self.P[MSCKF.Xi_idx, Xs_idx] = F @ self.P[MSCKF.Xi_idx, Xs_idx]
            self.P[Xs_idx, MSCKF.Xi_idx] = self.P[MSCKF.Xi_idx, Xs_idx].T        
        self.p_ts = ts_
        return self
    
    def propagate_trapezoidal(self, IMU0_, ts_):
        if np.all(IMU0_.fib_b) == 0:
            raise ValueError('IMU0 value is not inserted...')
        
        if self.pIMU.ts == 0: # initial euler propagation
            self.pIMU = IMU0_
            # self.propagate(IMU0_, ts_)
                
        Xs_idx = slice(MSCKF.Dxi, MSCKF.Dxi + MSCKF.Dxs*len(self.Xs))
        
        Pi = self.P[MSCKF.Xi_idx, MSCKF.Xi_idx]
        self.Xi, Pi, F = MSCKF.propagate_imu_trapezoidal(self.Xi, Pi, self.pIMU, IMU0_, self.Q, dt_=ts_ - self.p_ts)
        self.P[MSCKF.Xi_idx, MSCKF.Xi_idx] = Pi
        
        if self.P.shape[0] > MSCKF.Dxi:
            Xis = F @ self.P[MSCKF.Xi_idx, Xs_idx]
            self.P[MSCKF.Xi_idx, Xs_idx] = Xis
            self.P[Xs_idx, MSCKF.Xi_idx] = Xis.T                
        self.p_ts = ts_
        self.pIMU = IMU0_
        return self
    
    def correct(self, static_deadtracks_, dead_objects_, IMU0_, ts_):
        if (ts_ - self.p_ts) > 0:
            self.propagate(IMU0_, ts_)
            
        P = MSCKF.validate_cov_matrix(self.P)        
        Ho_s = Ho_o = np.empty((0, MSCKF.Dxi + MSCKF.Dxs*len(self.Xs)))
        ro_s = ro_o = np.empty(0)
        Ro_s = Ro_o = np.empty((0, 0))
        
        if static_deadtracks_:
            Ho_s, ro_s, Ro_s = MSCKF.generate_static_meas_model(self.Xs, P, self.Rs, self.Rpix, static_deadtracks_, self.lcam, self.rcam)     
        if dead_objects_:
            Ho_o, ro_o, Ro_o = MSCKF.generate_object_meas_model(self.Xs, P, self.Rs, self.Rd, self.Rpix, dead_objects_, self.lcam, self.rcam)       
        
        Ho = np.vstack((Ho_s, Ho_o))
        ro = np.hstack((ro_s, ro_o))
        Ro = np.block([[Ro_s, np.zeros((Ro_s.shape[0], Ro_o.shape[1]))],
                      [np.zeros((Ro_o.shape[0], Ro_s.shape[1])), Ro_o]])
            
        if ro.shape[0] > 10:
            Qho, Rho = np.linalg.qr(Ho)
            zero_row = np.all(Rho == 0, axis=1)
            Th = Rho[~zero_row, :]
            Q1 = Qho[:, ~zero_row]
            
            rn = Q1.T @ ro
            Rn = Q1.T @ Ro @ Q1
            
            Sn = Th @ P @ Th.T + Rn
            Kn = P @ Th.T @ np.linalg.inv(Sn)
            del_x_hat = Kn @ rn
            
            temp_mat = np.eye(len(del_x_hat)) - Kn @ Th
            P_hat = temp_mat @ P @ temp_mat.T + Kn @ Rn @ Kn.T
            P_hat = MSCKF.validate_cov_matrix(P_hat)
            
            if np.all(np.isreal(np.sqrt(np.diag(P_hat)))):
                Xs_idx = slice(MSCKF.Dxi, MSCKF.Dxi + MSCKF.Dxs*len(self.Xs))
                del_Xi = del_x_hat[MSCKF.Xi_idx]
                del_Xs = del_x_hat[Xs_idx]
                self.Xi = MSCKF.correct_Xi_from_del_Xi(self.Xi, del_Xi)
                self.Xs = MSCKF.correct_Xs_from_del_Xs(self.Xs, del_Xs)
                self.P = P_hat
                print('Successful sate correction !!!')               
            else:
                print('Failed state correction !!!')            
        return self
    
    def zupt(self):
        Nx = self.P.shape[0]
        
        Hz = np.zeros((3, Nx))
        Hz[:, MSCKF.v_idx] = np.eye(3)
        Rz = 0.001**2 * np.eye(3)
        
        S = Hz @ self.P @ Hz.T
        K = self.P @ Hz.T @ np.linalg.inv(S)
        r = -self.Xi.v
        del_x_hat = K @ r
        
        I_KH = np.eye(K.shape[0], Hz.shape[1]) - K @ Hz        
        P_hat = I_KH @ self.P @ I_KH.T + K @ Rz @ K.T       
        P_hat = MSCKF.validate_cov_matrix(P_hat)
        
        if np.all(np.isreal(np.sqrt(np.diag(P_hat)))):
            Xs_idx = slice(MSCKF.Dxi, MSCKF.Dxi + MSCKF.Dxs*len(self.Xs))
            del_Xi = del_x_hat[MSCKF.Xi_idx]            
            del_Xs = del_x_hat[Xs_idx]
            self.Xi = MSCKF.correct_Xi_from_del_Xi(self.Xi, del_Xi)
            self.Xs = MSCKF.correct_Xs_from_del_Xs(self.Xs, del_Xs)
            self.P = P_hat
        else:
            print('Failed state correction !!!')         
        return self
    
    def augment_state(self, frame):
        P = MSCKF.validate_cov_matrix(self.P)        
        Nmsckf = MSCKF.Dxi + MSCKF.Dxs * len(self.Xs)
        
        Js = np.zeros((MSCKF.Dxs, Nmsckf))
        Js[MSCKF.sp_idx, MSCKF.p_idx] = np.eye(3)
        Js[MSCKF.sq_idx, MSCKF.q_idx] = np.eye(3)
        
        I_aug = np.vstack((np.eye(Nmsckf), Js))
        P_aug = I_aug @ P @ I_aug.T
        
        Xs_aug = {}        
        Xs_aug['frame'] = frame
        Xs_aug['q'] = self.Xi.q
        Xs_aug['p'] = self.Xi.p                 
        if len(self.Xs) < self.max_slw:   
            self.Xs.append(Xs_aug)            
            self.P = P_aug
        else:
            self.Xs.pop(0)
            self.Xs.append(Xs_aug)          
            live_idx = np.hstack((np.arange(0, MSCKF.Dxi), np.arange(MSCKF.Dxi+MSCKF.Dxs, Nmsckf+MSCKF.Dxs)))
            self.P = P_aug[np.ix_(live_idx, live_idx)]            
        return self
            
    @ staticmethod    
    def propagate_imu_euler(Xi_, Pi_, IMU0_, Q_, dt_):
        fib_b0 = IMU0_.fib_b - Xi_.ba
        wib_b0 = IMU0_.wib_b - Xi_.bg
        
        # Attitude update
        theta_rvec = wib_b0*dt_   # rvec임.
        del_quat = Attitude.rvec2quat(theta_rvec)
        q_gb = Attitude.quatMultiply(Xi_.q, del_quat)
        
        # Acceleration coordinates transform
        Rgb0 = Attitude.quat2dcm(Xi_.q)
        Rgb1 = Attitude.quat2dcm(q_gb)
        Rb0b1 = Rgb0.T @ Rgb1
        dv0 = (1/2) * (fib_b0 + Rb0b1 @ fib_b0) * dt_
        dp0 = (1/2) * dv0 * dt_
        gg = Xi_.gg 
        
        # State update
        q = q_gb
        p = Xi_.p + Xi_.v * dt_ + Rgb0 @ dp0 + (1/2) * gg * dt_**2 
        v = Xi_.v + Rgb0 @ dv0 + gg * dt_
        ba = Xi_.ba 
        bg = Xi_.bg 
        
        Xi = NavState(q, p, v, ba, bg, gg)
        Phi = MSCKF.calc_state_transition_matrix(Xi_, fib_b0, dt_)
        Pi = Phi @ Pi_ @ Phi.T + Q_ * dt_        
        return Xi, Pi, Phi
        
    def propagate_imu_trapezoidal(Xi_, Pi_, IMU0_, IMU1_, Q_, dt_):        
        fib_b0 = IMU0_.fib_b - Xi_.ba
        fib_b1 = IMU1_.fib_b - Xi_.ba
        wib_b0 = IMU0_.wib_b - Xi_.bg        
        wib_b1 = IMU1_.wib_b - Xi_.bg
        
        # Attitude update
        q_gb = Attitude.quat_integrate_RK4(Xi_.q, wib_b0, wib_b1, dt_)
        
        # Acceleration coordinates transform
        Rgb0 = Attitude.quat2dcm(Xi_.q)
        Rgb1 = Attitude.quat2dcm(q_gb)
        Rb0b1 = Rgb0.T @ Rgb1
        dv0 = (1/2) * (fib_b0 + Rb0b1 @ fib_b1) * dt_
        dp0 = (1/2) * dv0 * dt_
        gg = Xi_.gg
        
        # State update
        q = q_gb
        p = Xi_.p + Xi_.v * dt_ + Rgb0 @ dp0 + (1/2) * gg * dt_**2
        v = Xi_.v + Rgb0 @ dv0 + gg * dt_
        ba = Xi_.ba 
        bg = Xi_.bg 
        
        Xi = NavState(q, p, v, ba, bg, gg)
        Phi = MSCKF.calc_state_transition_matrix(Xi_, fib_b0, dt_)
        Pi = Phi @ Pi_ @ Phi.T + Q_ * dt_
        return Xi, Pi, Phi

    def calc_state_transition_matrix(Xi_, fib_b_, dt_):
        Rgb = Attitude.quat2dcm(Xi_.q)
        F = np.zeros((15, 15))
        F[MSCKF.q_idx, MSCKF.bg_idx] = -Rgb
        F[MSCKF.p_idx, MSCKF.v_idx]  = np.eye(3, 3)
        F[MSCKF.v_idx, MSCKF.q_idx]  = -SO3.skew(Rgb@fib_b_)
        F[MSCKF.v_idx, MSCKF.ba_idx] = -Rgb
        Phi = np.eye(15, 15) + F * dt_ + (1/2) * F @ F * dt_ * dt_
        return Phi
    
    def validate_cov_matrix(P_):
        P_pd = (1/2) * (P_ + P_.T)          # x' * P_pd * x > 0 : positive definite
        try:
            np.linalg.cholesky(P_pd)
        except np.linalg.LinAlgError:       # if not positive
            eigvals, eigvecs = np.linalg.eig(P_pd)
            
            eigvals = np.real(eigvals)
            eigvals[eigvals < np.finfo(float).eps] = 1e-12
            
            P_pd = np.dot(np.dot(eigvecs, np.diag(eigvals)), eigvecs.T)
        if np.iscomplexobj(P_pd):
            P_pd = P_pd.real.astype(np.float64)
            
        return P_pd
        
    def generate_static_meas_model(Xs_, P_, Rs_, Rpix_, static_deadtracks_, lcam_, rcam_):
        """Generate point reprojection measurement model 
        Args:
            Xs_ (list): dict with q, p, frame. 
            P_ (np.array): error-state covariance
            R_ (np.array): measurement noise covariance
            deadtracks_ (list): list - dict(feature_id, frame, pts, un_pts) -> tracks
            lcam_ (dict): lcam params
            rcam_ (dict): rcam params
        Returns:
            Ho, ro, Ro (np.array): Left-projected Jacobian, measurement residual, measurement noise covariance
        """
        # Static measurements (original deadtracks)
        N_stracks = len(static_deadtracks_)
        Ho_s = np.empty((0, MSCKF.Dxi + MSCKF.Dxs*len(Xs_)))
        ro_s = np.empty(0)
        Ro_s = np.empty((0, 0))        
        
        for ti in range(N_stracks):
            track = static_deadtracks_[ti]
            track = MSCKF.triangulate_GN(track, Xs_, Rpix_, lcam_, rcam_)          
            if track['bad_status']:
                print(f'Track {ti} : bad triangulation')
            else:
                Hx_j, Hf_j = MSCKF.calc_stereo_meas_jacobian(track, Xs_, lcam_, rcam_)
                r_j = track['residuals']
            
                r_j = (r_j.T).flatten()
                val_idx = ~np.isnan(r_j)
                r_j = r_j[val_idx] 
                Hx_j = Hx_j[val_idx, :]
                Hf_j = Hf_j[val_idx, :]
                R_j = np.diag(np.tile(np.diag(Rs_), int(len(r_j)/4)))                
                Lj = scipy.linalg.null_space(Hf_j.T)            
                if Lj.shape[0] != 0:
                    Ho_j = Lj.T @ Hx_j
                    ro_j = Lj.T @ r_j
                    Ro_j = Lj.T @ R_j @ Lj
                    
                    gamma = ro_j.T @ np.linalg.inv(Ho_j @ P_ @ Ho_j.T + Ro_j) @ ro_j
                    if gamma < chi2.ppf(MSCKF.chi_trust, len(ro_j)):
                        Ho_s = np.vstack((Ho_s, Ho_j))
                        ro_s = np.hstack((ro_s, ro_j))
                        Ro_s = np.block([[Ro_s, np.zeros((Ro_s.shape[0], Ro_j.shape[1]))],
                                    [np.zeros((Ro_j.shape[0], Ro_s.shape[1])), Ro_j]])
                    else:
                        print(f'Track {ti} : Failed to pass Chi-square test !!!')
                        pass                  
        return Ho_s, ro_s, Ro_s
    
    def generate_dynamic_meas_model(Xs_, P_, Rd_, Rpix_, dynamic_deadtracks_, lcam_, rcam_):
        """Generate point reprojection measurement model 
        Args:
            Xs_ (list): dict with q, p, frame. 
            P_ (np.array): error-state covariance
            R_ (np.array): measurement noise covariance
            deadtracks_ (list): list - dict(feature_id, frame, pts, un_pts) -> tracks
            lcam_ (dict): lcam params
            rcam_ (dict): rcam params
        Returns:
            Ho, ro, Ro (np.array): Left-projected Jacobian, measurement residual, measurement noise covariance
        """
        # Dynamic measurements (DFMM)
        N_dtracks = len(dynamic_deadtracks_)
        Ho_d = np.empty((0, MSCKF.Dxi + MSCKF.Dxs*len(Xs_)))
        ro_d = np.empty(0)
        Ro_d = np.empty((0, 0))
        
        for ti in range(N_dtracks):
            track = dynamic_deadtracks_[ti]  
            Hx_j, Hf_j = MSCKF.calc_dynamic_meas_jacobian(track, Xs_, lcam_, rcam_)
            r_j = track['residuals'][:, 1:]        
            r_j = (r_j.T).flatten()
            val_idx = ~np.isnan(r_j)
            r_j = r_j[val_idx] 
            Hx_j = Hx_j[val_idx, :]
            Hf_j = Hf_j[val_idx, :]
            R_j = np.diag(np.tile(np.diag(Rd_), int(len(r_j)/4)))                
            Lj = scipy.linalg.null_space(Hf_j.T)            
            if Lj.shape[0] != 0:
                Ho_j = Lj.T @ Hx_j
                ro_j = Lj.T @ r_j
                Ro_j = Lj.T @ R_j @ Lj
                
                gamma = ro_j.T @ np.linalg.inv(Ho_j @ P_ @ Ho_j.T + Ro_j) @ ro_j
                if gamma < chi2.ppf(MSCKF.chi_trust, len(ro_j)):
                    Ho_d = np.vstack((Ho_d, Ho_j))
                    ro_d = np.hstack((ro_d, ro_j))
                    Ro_d = np.block([[Ro_d, np.zeros((Ro_d.shape[0], Ro_j.shape[1]))],
                                [np.zeros((Ro_j.shape[0], Ro_d.shape[1])), Ro_j]])
                else:
                    print(f'Track {ti} : Failed to pass Chi-square test !!!')
                    pass                  
            
        return Ho_d, ro_d, Ro_d
                 
    def generate_object_meas_model(Xs_, P_, Rs_, Rd_, Rpix_, dead_objects_, lcam_, rcam_):        
        Ho_so = Ho_do = np.empty((0, MSCKF.Dxi + MSCKF.Dxs*len(Xs_)))
        ro_so = ro_do = np.empty(0)
        Ro_so = Ro_do = np.empty((0, 0))
        
        static_object_feas, dynamic_object_feas = MSCKF.process_dead_objects(Xs_, dead_objects_, lcam_, rcam_)
        
        if static_object_feas:
            Ho_so, ro_so, Ro_so = MSCKF.generate_static_meas_model(Xs_, P_, Rs_, Rpix_, static_object_feas, lcam_, rcam_)
            
        if dynamic_object_feas:
            Ho_do, ro_do, Ro_do = MSCKF.generate_dynamic_meas_model(Xs_, P_, Rd_, Rpix_, dynamic_object_feas, lcam_, rcam_)
            
        Ho_o = np.vstack((Ho_so, Ho_do))
        ro_o = np.hstack((ro_so, ro_do))
        Ro_o = np.block([[Ro_so, np.zeros((Ro_so.shape[0], Ro_do.shape[1]))],
                        [np.zeros((Ro_do.shape[0], Ro_so.shape[1])), Ro_do]])
            
        return Ho_o, ro_o, Ro_o
        
    def process_dead_objects(Xs_, dead_objects_, lcam_, rcam_):
        slw_frames = np.zeros(len(Xs_))      
        for i in range(len(Xs_)):
            slw_frames[i] = Xs_[i]['frame']  
            
        dynamic_object_feas = []
        static_object_feas = []
                
        for dead_object in dead_objects_:                            
            object_id = list(dead_object.keys())[0]
            object_values = dead_object[object_id]            
            
            view_ids = [key for key in object_values.keys() if isinstance(key, (int, float))]              
            feature_ids = object_values[view_ids[0]][0, :]
            
            N_views = len(view_ids)
            Nf = len(feature_ids)                       

            # Dynamic objects
            if object_values['dyna_state']:                        
                # Calculate intersected Tlil0                
                _, slw_idx, view_idx = np.intersect1d(slw_frames, view_ids, return_indices=True)            
                Tlil0_set = np.zeros((4, 4, N_views))
                for i in range(N_views):
                    xs = Xs_[slw_idx[i]]
                    Tgli = np.block([[Attitude.quat2dcm(xs['q']), xs['p'].reshape(-1, 1)],
                                    [0, 0, 0, 1]]) @ lcam_['Tbc']
                    if i == 0:
                        Tgl0 = Tgli
                    Tlil0_set[:, :, i] = np.linalg.inv(Tgl0) @ Tgli # Tl0li
                                
                residuals = np.zeros((4, Nf, N_views))                      # (4 x Nf x Ntracks)
                fo_l0 = object_values[view_ids[0]][9:12:,:]
                fo_0_g = Tgl0[:3, :3] @ fo_l0 + Tgl0[:3, 3].reshape(-1, 1)
                
                for i in range(N_views)[1:]:
                    fo_Ik = object_values[view_ids[i]][5:9:,:]          # un_uv : 4 x Nf
                    fo_lk = object_values[view_ids[i]][9:12:,:]         # fo_lk : 3 x Nf
                    
                    fo_lk_hat = Tlil0_set[:3, :3, i] @ fo_l0 + Tlil0_set[3, :3, i].reshape(-1, 1)
                    R, t, fo_lk_hat_OMC = MSCKF.estimate_object_motion(fo_lk, fo_lk_hat)
                    fo_rk_hat_OMC = lcam_['Trl'][:3, :3] @ fo_lk_hat_OMC + lcam_['Trl'][3, :3].reshape(-1, 1)
                    
                    fo_Ik_l_hat = (lcam_['K'] @ (fo_lk_hat_OMC / fo_lk_hat_OMC[2, :]))[:2, :]
                    fo_Ik_r_hat = (rcam_['K'] @ (fo_rk_hat_OMC / fo_rk_hat_OMC[2, :]))[:2, :]
                    
                    # New implementation
                    Tlig = Tlil0_set[:, :, i] @ np.linalg.inv(Tgl0)
                    Trig = lcam_['Trl'] @ Tlig
                    Tgli = np.linalg.inv(Tlig)
                    
                    fo_k_g = Tgli[:3, :3] @ fo_lk + Tgli[:3, 3].reshape(-1, 1)                    
                    R, t, fo_k_g_hat =  MSCKF.estimate_object_motion(fo_k_g, fo_0_g)
                    
                    fo_lk_hat = Tlig[:3, :3] @ fo_k_g_hat + Tlig[:3, 3].reshape(-1, 1)
                    fo_rk_hat = Trig[:3, :3] @ fo_k_g_hat + Trig[:3, 3].reshape(-1, 1)
                    
                    fo_Ik_l_hat_ =  (lcam_['K'] @ (fo_lk_hat / fo_lk_hat[2, :]))[:2, :]
                    fo_Ik_r_hat_ =  (lcam_['K'] @ (fo_rk_hat / fo_rk_hat[2, :]))[:2, :]
                    
                    residuals[:, :, i] = fo_Ik - np.vstack((fo_Ik_l_hat_, fo_Ik_r_hat_))
                    
                    # residuals[:, :, i] = fo_Ik - np.vstack((fo_Ik_l_hat, fo_Ik_r_hat)) : Original
                
                # Mapping
                for j in range(Nf):
                    temp_dynamic_feas = {}                    
                    temp_dynamic_feas['feature_id']   = feature_ids[j]    
                    temp_dynamic_feas['frame']        = view_ids
                    # temp_dynamic_feas['pts']          = np.zeros((4, len(view_ids)))
                    # temp_dynamic_feas['un_pts']       = np.zeros((4, len(view_ids)))
                    temp_dynamic_feas['f_li']         = np.zeros((3, N_views))
                    temp_dynamic_feas['f_g']          = np.zeros((3, N_views))
                    temp_dynamic_feas['residuals']     = np.zeros((4, N_views))                                          # 4 x Nslw 형태로 tracks['residuals'] 제공할 것 !!!!!
                    for i in range(N_views):        
                        # temp_dynamic_feas['pts'][:, i]         = object_values[view_ids[i]][1:5, j]
                        # temp_dynamic_feas['un_pts'][:, i]      = object_values[view_ids[i]][5:9, j]                
                        temp_dynamic_feas['f_li'][:, i]        = object_values[view_ids[i]][9:12, j]                
                        temp_dynamic_feas['f_g'][:, i]         = object_values[view_ids[i]][12:, j]                
                        temp_dynamic_feas['residuals'][:, i]    = residuals[:, j, i]                                            # 4 x Nslw 형태로 tracks['residuals'] 제공할 것 !!!!!
                    dynamic_object_feas.append(temp_dynamic_feas)
            # Static objects
            else:
                for j in range(Nf):
                    temp_static_feas = {}
                    temp_static_feas['feature_id']   = feature_ids[j]
                    temp_static_feas['frame']        = view_ids
                    temp_static_feas['pts']          = np.empty((4, len(view_ids)))
                    temp_static_feas['un_pts']       = np.empty((4, len(view_ids)))
                    for i in range(len(view_ids)):            
                        temp_static_feas['pts'][:,i]      = object_values[view_ids[i]][1:5, j]
                        temp_static_feas['un_pts'][:,i]   = object_values[view_ids[i]][5:9, j]
                    static_object_feas.append(temp_static_feas)           
        return static_object_feas, dynamic_object_feas
                   
    def triangulate_GN(track_, Xs_, Rpix_, lcam_, rcam_):
        """Triangulation with multi-view / Nonlinear optimization using Gauss-Newton method
        Args:
            track_ (dict): _description_
            Xs_ (_type_): _description_
            Rpix_ (float): pixel noise covariance
            lcam_ (_type_): _description_
            rcam_ (_type_): _description_
        Returns:
            deadtrack (dict): pf_g, pf_l0, residual, sliding window id, reprojected points
        """
        Rpix_ = 2   # fixed constant
        Ntracks = len(track_['frame'])
        Nslw = len(Xs_)
        
        Xs_frame = np.zeros(Nslw)
        for i in range(Nslw):
            Xs_frame[i] = Xs_[i]['frame']
        intersect_frame, s_idx, f_idx = np.intersect1d(Xs_frame, track_['frame'], return_indices=True)
        
        Tset = np.zeros((4, 4, Ntracks))
        for i in range(Ntracks):
            xs = Xs_[s_idx[i]]
            Tgli = np.block([[Attitude.quat2dcm(xs['q']), xs['p'].reshape(-1, 1)],
                              [0, 0, 0, 1]]) @ lcam_['Tbc']
            if i == 0:
                Tgl0 = Tgli
            Tset[:, :, i] = np.linalg.inv(Tgl0) @ Tgli # Tl0li
                
        two_view_pts, right_idx = MSCKF.choice_two_points(track_)
        Tl0ri = Tset[:, :, right_idx] @ lcam_['Tlr']        # Tl0ri
        pf_l0_0, bad_status_init  = MSCKF.triangulate_two_view(Tl0ri, lcam_, rcam_, two_view_pts)
        
        # two_view_pts = track_['un_pts'][:, 0].reshape(2,2).T
        # Tl0r0 = Tset[:, :, 0] @ lcam_['Tlr']  # 1st left and right cam pose  
        # pf_l0_0, bad_status_init  = MSCKF.triangulate_two_view(Tl0r0, lcam_, rcam_, two_view_pts)
        pf_l0, bad_status         = MSCKF.triangulate_multi_stereo_view(track_['un_pts'], Tset, pf_l0_0, lcam_, rcam_, Rpix_)
        
        if bad_status:
            stop = 1
        pf_g = Tgl0 @ np.hstack((pf_l0, 1)).reshape(-1, 1)
        pf_g = pf_g[0:3]
        
        # Residuals
        reproj_pts = np.zeros((4, Ntracks))
        residuals = np.zeros((4, Ntracks))
        for i in range(Ntracks):
            # if z = [u, v]
            # lcam
            Tl0li = Tset[:, :, i]
            pf_li = np.linalg.inv(Tl0li) @ np.hstack((pf_l0, 1)).reshape(-1, 1)
            pf_li = pf_li[:3]/pf_li[3]  # [X, Y, Z]^li
            pf_I_li = lcam_['K'] @ np.vstack((pf_li[:2]/pf_li[2], 1))            
            # rcam
            T0i_r = Tset[:, :, i] @ lcam_['Tlr']
            pf_ri = np.linalg.inv(T0i_r) @ np.hstack((pf_l0, 1)).reshape(-1, 1)
            pf_ri = pf_ri[:3]/pf_ri[3]  # [X, Y, Z]^li      
            pf_I_ri = rcam_['K'] @ np.vstack((pf_ri[:2]/pf_ri[2], 1))   
            
            reproj_pts[:, i] = np.vstack([pf_I_li[:2], pf_I_ri[:2]]).reshape(1, -1)
            residuals[:, i] = track_['un_pts'][:, i] - reproj_pts[:, i]
            
        deadtrack = track_
        deadtrack['pf_g'] = pf_g        
        deadtrack['residuals'] = residuals        
        deadtrack['bad_status'] = bad_status
        deadtrack['s_idx'] = s_idx                
        return deadtrack
        
    def triangulate_two_view(Tc0c1_, lcam_, rcam_, un_pts_):
        """Triangulation with two-view
            Eq : p_c = lambda[0]*v_0 = R01 @ (lambda[1]*v_1) + t01_0
            [v_0, -R01@v_1] @ lambda = t01_0
            Z : depth
            v_0, v_1 : normalized pf_c0, pf_c1
        Args:
            T01_ (_type_): Transformation matrix
            lcam_ (dict): lcam_params
            rcam_ (dict): rcam_params
            un_pts_ (np.array): 2x2, [u_l, u_r; v_l, v_r], un_uv
        Returns:
            f_c0 : f at {cam 0}
            bad_status_init : 
        """
        Rc0c1 = Tc0c1_[:3, :3]        
        tc0c1 = Tc0c1_[:3, 3]
        pf_c0_n = np.linalg.solve(a=lcam_['K'], b=np.hstack((un_pts_[:, 0], 1))) # A 가 정방행렬일 때, x = A^(-1) @ b 푸는법
        pf_c1_n = np.linalg.solve(a=rcam_['K'], b=np.hstack((un_pts_[:, 1], 1))) 
        A = np.hstack((pf_c0_n.reshape(-1, 1), (-Rc0c1 @ pf_c1_n).reshape(-1, 1)))
        b = tc0c1        
        Z = np.linalg.pinv(A) @ b                 # (A^T@A)^-1 @ A^T @ b : Least-square sol.        
        f_c0 = Z[0] * pf_c0_n
        bad_status_init = False
        if Z[0] < 0:
            bad_status_init = True        
        return f_c0, bad_status_init
    
    def triangulate_stereo_view(lcam_, rcam_, un_pts_):
        """Traingulate with stereo vision
        Args:
            lcam_ (dict): lcam params
            rcam_ (dict): rcam params
            un_pts_ (np.array): 4xNf, undistorted points at image plane, un_uv 
        Returns:
            f_l: points at left camera, {l}
        """        
        Nf = un_pts_.shape[1]
        f_Il = np.vstack((un_pts_[:2, :], np.ones(Nf)))      # uv_l
        f_Ir = np.vstack((un_pts_[2:, :], np.ones(Nf)))      # uv_r
        nf_l = np.linalg.inv(lcam_['K']) @ f_Il                         # nf_l = [X/Z, Y/Z, 1] = K^-1 @ [u, v, 1], 3xNf     
        nf_l = nf_l/nf_l[2,:] 
        
        disparity = f_Il[0, :] - f_Ir[0, :]                              # d = u_l - u_r, 1xNf
        Z = lcam_['K'][0, 0] * lcam_['Tlr'][0, 3] / disparity
        val_idx = np.where(Z > 0)[0]
        f_l = Z*nf_l   
        f_l = f_l[:, val_idx]                   
        return f_l, val_idx
    
    def triangulate_multi_stereo_view(un_pts_, Tset_, f_l0_0_, lcam_, rcam_, Rpix_):
        """ Triangulate with multi stereo view with Gauss-Newton method
        Args:
            un_pts_ (np.array): point feature at i-th image plane, pf_I_li = [u, v]        
            Tset_ (np.array): Tl0li
            f_l0_0_ (np.array): initial value of point feature at 0-th left camera, pf_l0 = [X, Y, Z]
            lcam_ (dict): left cam params
            rcam_ (dict): right cam params
            R_ (np.array): covariance
        Returns:
            f_l0 (np.array): converged value of point feature at 0-th left camera, pf_l0 = [X, Y, Z]
        """
        Ntracks = Tset_.shape[2]
        x_hat = np.hstack((f_l0_0_[:2]/f_l0_0_[2], 1/f_l0_0_[2]))    # [X/Z, Y/Z, 1/Z]
        max_iter = 10
        C_prev = 0
        
        for i in range(max_iter):
            A = np.zeros((4*Ntracks, 3))                                # Set memory allocation to solve Ax = b
            b = np.zeros((4*Ntracks, 1))
            W = np.zeros((4*Ntracks, 4*Ntracks))
            
            for j in range(Ntracks):
                Tlil0 = np.linalg.inv(Tset_[:, :, j])                   # T^li_l0
                Tril0 = lcam_['Trl'] @ Tlil0                            # T^ri_l0
                
                h_l = (Tlil0 @ np.hstack((x_hat[:2], 1, x_hat[2])).reshape(-1, 1)).flatten()
                h_r = (Tril0 @ np.hstack((x_hat[:2], 1, x_hat[2])).reshape(-1, 1)).flatten()
                
                r_idx = slice(4*j, 4*(j + 1))
                
                # z = pf_I_li = [u v]
                z_hat_l = lcam_['K'] @ (np.hstack((h_l[:2]/h_l[2], 1)).reshape(-1, 1))
                z_hat_r = rcam_['K'] @ (np.hstack((h_r[:2]/h_r[2], 1)).reshape(-1, 1))
                z_hat = np.vstack((z_hat_l[:2], z_hat_r[:2]))
                
                b[r_idx] = un_pts_[:, j].reshape(-1, 1) - z_hat
                
                # Jacobian     # g = pf_li 
                dh_dg = lcam_['K'][:2, :2] @ np.vstack(([1/h_l[2],   0,          -h_l[0]/(h_l[2]**2)],
                                                        [0,          1/h_l[2],   -h_l[1]/(h_l[2]**2)]))
                dg_dx = Tlil0[:3, [0, 1, 3]]
                AblockL = dh_dg @ dg_dx
                
                dh_dg = rcam_['K'][:2, :2] @ np.vstack([[1/h_r[2],   0,          -h_r[0]/h_r[2]**2],
                                                        [0,          1/h_r[2],   -h_r[1]/h_r[2]**2]])
                dg_dx = Tril0[:3, [0, 1, 3]]
                AblockR = dh_dg @ dg_dx                
                Ablock = np.vstack((AblockL, AblockR))
                A[r_idx, :] = Ablock
                
            #  Gauss-Newton update            
            C_new = (1/2) * b.flatten() @ b / Rpix_
            AtA = A.T @ A / Rpix_
            dx_star = np.linalg.inv(AtA + 1e-3 * np.diag(np.diag(AtA))) @ A.T @ b / Rpix_       # dx* = (A^T*W*A + eI_3) * A^T*W*b
            x_hat = x_hat + dx_star.flatten()                                                   # x^hat+ = x^hat- + dx*            
            del_C = np.abs((C_new - C_prev) / C_new)
            C_prev = C_new
            
            if del_C < 1e-6:
                break       
        
        f_l0 = np.hstack((x_hat[:2]/x_hat[2], 1/x_hat[2]))
        bad_status = False
        
        if (1/x_hat[2] < 0.1) or (1/x_hat[2] > 50) or (C_new[0] > Ntracks*10):
            bad_status = True         
        return f_l0, bad_status
    
    def calc_stereo_meas_jacobian(track_, Xs_, lcam_, rcam_):
        N_tracks = len(track_['s_idx'])
        N_slw = len(Xs_)
        
        Hx_j = np.zeros((4*N_tracks, MSCKF.Dxi + MSCKF.Dxs * N_slw))
        Hf_j = np.zeros((4*N_tracks, 3))
        
        for i in range(N_tracks):
            pose = Xs_[track_['s_idx'][i]]
            Tgbi = np.block([[Attitude.quat2dcm(pose['q']), pose['p'].reshape(-1, 1)],
                              [0, 0, 0, 1]])
            Tgli = Tgbi @ lcam_['Tbc']
            Tgri = Tgli @ lcam_['Tlr']
            pf_li = np.linalg.inv(Tgli) @ np.vstack((track_['pf_g'], 1))
            pf_ri = np.linalg.inv(Tgri) @ np.vstack((track_['pf_g'], 1))
                        
            r_idx = slice(4*i, 4*(i+1))
            s_idx = MSCKF.Dxi + MSCKF.Dxs*(track_['s_idx'][i])   
            
            # if z = [u v]
            Jp_li_j = lcam_['K'][:2, :2] @ np.block([[1/pf_li[2], 0, -pf_li[0]/pf_li[2]**2],
                                                     [0, 1/pf_li[2], -pf_li[1]/pf_li[2]**2]])
            Jp_ri_j = rcam_['K'][:2, :2] @ np.block([[1/pf_ri[2], 0, -pf_ri[0]/pf_ri[2]**2],
                                                     [0, 1/pf_ri[2], -pf_ri[1]/pf_ri[2]**2]]) 
            
            # if z = [X/Z Y/Z]
            # Jp_li_j = np.block([[1/pf_li[2], 0, -pf_li[0]/pf_li[2]**2],
            #                     [0, 1/pf_li[2], -pf_li[1]/pf_li[2]**2]])
            # Jp_ri_j = np.block([[1/pf_ri[2], 0, -pf_ri[0]/pf_ri[2]**2],
            #                     [0, 1/pf_ri[2], -pf_ri[1]/pf_ri[2]**2]]) 
            
            Hf_li_j = Jp_li_j @ Tgli[:3, :3].T
            Hf_ri_j = Jp_ri_j @ Tgri[:3, :3].T                        
            Hf_j[r_idx, :] = np.vstack((Hf_li_j, Hf_ri_j))
                       
            sq_idx = slice(s_idx, s_idx + 3)
            sp_idx = slice(s_idx + 3, s_idx + 6)
            Hx_j[r_idx, sq_idx] = np.vstack(((Hf_li_j @ SO3.skew(track_['pf_g'].flatten() - Tgbi[:3,3])),
                                             (Hf_ri_j @ SO3.skew(track_['pf_g'].flatten() - Tgbi[:3,3]))))            
            Hx_j[r_idx, sp_idx] = np.vstack((-Hf_li_j,
                                             -Hf_ri_j))            
        return Hx_j, Hf_j 

    def calc_dynamic_meas_jacobian(track_, Xs_, lcam_, rcam_):                
        N_slw = len(Xs_)
        Xs_frame = np.zeros(N_slw)
        for i in range(N_slw):
            Xs_frame[i] = Xs_[i]['frame']
        _, s_idx, f_idx = np.intersect1d(Xs_frame, track_['frame'], return_indices=True)
        
        N_tracks = len(s_idx)   # intersected slws
        
        Hx_j = np.zeros((4*(N_tracks - 1), MSCKF.Dxi + MSCKF.Dxs * N_slw))
        Hf_j = np.zeros((4*(N_tracks - 1), 3))
        
        for i in range(N_tracks)[1:]: # without the 0-th pose : residual of 0-th pose is 0 because we don't know the absolute feature position in the global frame
            pose = Xs_[s_idx[i]]
            Tgbi = np.block([[Attitude.quat2dcm(pose['q']), pose['p'].reshape(-1, 1)],
                              [0, 0, 0, 1]])
            Tgli = Tgbi @ lcam_['Tbc']
            Tgri = Tgli @ lcam_['Tlr']
            
            # f_li = np.linalg.inv(Tgli) @ np.hstack((track_['f_g'][:, i], 1)).reshape(-1, 1)
            f_li = track_['f_li'][:, i]
            f_ri = lcam_['Trl'][:3, :3] @ f_li + lcam_['Trl'][:3, 3]
            f_g = Tgli[:3, :3] @ f_li + Tgli[:3, 3]
                        
            r_idx = slice(4*(i-1), 4*i)
            s_idx_ = MSCKF.Dxi + MSCKF.Dxs*(s_idx[i])   
            
            # if z = [u v]
            Jp_li_j = lcam_['K'][:2, :2] @ np.block([[1/f_li[2], 0, -f_li[0]/f_li[2]**2],
                                                     [0, 1/f_li[2], -f_li[1]/f_li[2]**2]])
            Jp_ri_j = rcam_['K'][:2, :2] @ np.block([[1/f_ri[2], 0, -f_ri[0]/f_ri[2]**2],
                                                     [0, 1/f_ri[2], -f_ri[1]/f_ri[2]**2]]) 
            
            Hf_li_j = Jp_li_j @ Tgli[:3, :3].T
            Hf_ri_j = Jp_ri_j @ Tgri[:3, :3].T                        
            Hf_j[r_idx, :] = np.vstack((Hf_li_j, Hf_ri_j))
                       
            sq_idx = slice(s_idx_, s_idx_ + 3)
            sp_idx = slice(s_idx_ + 3, s_idx_ + 6)
            Hx_j[r_idx, sq_idx] = np.vstack(((Hf_li_j @ SO3.skew(f_g.flatten() - Tgbi[:3,3])),
                                             (Hf_ri_j @ SO3.skew(f_g.flatten() - Tgbi[:3,3]))))    
            # Hx_j[r_idx, sq_idx] = np.vstack(((Hf_li_j @ SO3.skew(track_['f_g'][:, i].flatten() - Tgbi[:3,3])),
            #                                  (Hf_ri_j @ SO3.skew(track_['f_g'][:, i].flatten() - Tgbi[:3,3]))))            
            Hx_j[r_idx, sp_idx] = np.vstack((-Hf_li_j,
                                             -Hf_ri_j))            
        return Hx_j, Hf_j 
    
    def correct_Xi_from_del_Xi(Xi_, del_Xi_):        
        del_q = Attitude.rvec2quat(del_Xi_[MSCKF.q_idx])
        del_p = del_Xi_[MSCKF.p_idx]
        del_v = del_Xi_[MSCKF.v_idx]
        del_ba = del_Xi_[MSCKF.ba_idx]
        del_bg = del_Xi_[MSCKF.bg_idx]
        del_Nav = NavState(del_q, del_p, del_v, del_ba, del_bg, np.zeros(3))
        
        Xi = Xi_.update_from_global_del_Nav(del_Nav)
        return Xi
        
    def correct_Xs_from_del_Xs(Xs_, del_Xs_):
        Nslw = len(Xs_)
        Xs = []
        for i in range(Nslw):
            Xs_q_idx = slice(MSCKF.Dxs*i, MSCKF.Dxs*i + 3)
            Xs_p_idx = slice(MSCKF.Dxs*i + 3, MSCKF.Dxs*i + 6)            
            
            del_q = Attitude.rvec2quat(del_Xs_[Xs_q_idx])
            del_p = del_Xs_[Xs_p_idx]
            
            Xs_corrected = {'q' : Attitude.quatMultiply(del_q, Xs_[i]['q']), 'p' : Xs_[i]['p'] + del_p, 'frame' : Xs_[i]['frame']}
            Xs.append(Xs_corrected)            
        return Xs
            
    def choice_two_points(track_):  
        """Choice two farthest points for initial value of GN triangulation        """      
        dist = 0
        f_I0_l = track_['un_pts'][:2, 0]     
        for j, f_Ii_r in enumerate(track_['un_pts'][2:4, :].T):
            new_dist = np.linalg.norm(f_I0_l - f_Ii_r)
            if new_dist > dist:
                dist = new_dist
                two_view_points = np.hstack((f_I0_l.reshape(-1, 1), f_Ii_r.reshape(-1, 1)))
                right_idx = j
        
        return two_view_points, right_idx
    
    def estimate_object_motion(f1_l1_, f0_l1_):
        """Estimate object motion based on PCL registration with known correspondences
        Args:
            f0_l1 (np.array): feature points at [0] in {l1}
            f1_l1 (np.array): feature points at [1] in {l1}            
        Returns:
            R, t, f0_l1_hat : rotation, translation, feas compensated with Object Motion
        """

        mu_f0 = np.mean(f0_l1_, axis=1)        
        mu_f1 = np.mean(f1_l1_, axis=1)
        
        df_okm1 = f0_l1_ - mu_f0.reshape(-1,1)
        df_ok   = f1_l1_ - mu_f1.reshape(-1,1)

        sig_fokm1 = np.mean(np.sum(df_okm1**2, axis=0))
        sig_fok = np.mean(np.sum(df_ok**2, axis=0))
        
        Sig = df_ok @ df_okm1.T/df_ok.shape[1]

        U, D, VT = np.linalg.svd(Sig)
        
        if np.linalg.det(U) * np.linalg.det(VT.T) < 0:
            W = np.diag([1, 1, -1])
        else:
            W = np.eye(3)
        
        R = U @ W @ VT      
        t = mu_f1 - R @ mu_f0                
        f0_l1_hat = R @ f0_l1_ + t.reshape(-1,1)       

        return R, t, f0_l1_hat
        