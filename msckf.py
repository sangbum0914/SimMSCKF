from utils.attitude import Attitude
from utils.nav_state import NavState
from utils.SO3 import SO3

from scipy.stats import chi2
import numpy as np
import scipy.linalg

from concurrent.futures import ProcessPoolExecutor
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
    std_na = 0.23e-3/9.81;                  # [m/s^2]     : mg                  
    std_ng = 0.0135*d2r;                    # [rad/sec]   : deg/hr
    std_wa = 0.25e-3*9.81/np.sqrt(100);     # [m/s^4]     : mg/sqrt(Hz)   
    std_wg = (14.5*d2r)/3600/np.sqrt(100);    # [rad/sec^3] : deg/s/sqrt(Hz)
    
    def __init__(self, Ximu_, Xslw_, min_slw_, max_slw_, lcam_, rcam_):
        self.Xi = Ximu_
        self.Xs = Xslw_             # list 
        self.min_slw = min_slw_
        self.max_slw = max_slw_
        self.lcam = lcam_
        self.rcam = rcam_
            
    def get_initial_uncertainty():            
        std_q = 0.1*MSCKF.d2r
        std_p = 0.01
        std_v = 0.1  
        std_ba = 20e-3*9.81;                          # [m/s^2] : 20 mg
        std_bg = 0.5*MSCKF.d2r;                       # [rad/s] : 0.5 deg/s
        
        P0 = np.diag(np.hstack([std_q**2*np.ones(3), std_p**2*np.ones(3), std_v**2*np.ones(3), std_ba**2*np.ones(3), std_bg**2*np.ones(3)]))
        Q = np.diag(np.hstack([MSCKF.std_na**2*np.ones(3), np.zeros(3), MSCKF.std_ng**2*np.ones(3), MSCKF.std_wa**2*np.ones(3), MSCKF.std_wg**2*np.ones(3)]))        
        return P0, Q
    
    def set_uncertainty(self, P_, Q_, Rpix_):
        self.P = P_
        self.Q = Q_
        self.Rpix = Rpix_
        self.Rf = Rpix_ * np.eye(4) 
        
        return self
        
    def set_result(self, Nsim_):
        self.est = {'q' : np.zeros([4, Nsim_]), 'p' : np.zeros([3, Nsim_]), 'v' : np.zeros([3, Nsim_]), 'ba' : np.zeros([3, Nsim_]), 'bg' : np.zeros([3, Nsim_])}        
        self.std = {'q' : np.zeros([3, Nsim_]), 'p' : np.zeros([3, Nsim_]), 'v' : np.zeros([3, Nsim_]), 'ba' : np.zeros([3, Nsim_]), 'bg' : np.zeros([3, Nsim_])}        
        
        self.est['q'][:, 0] = self.Xi.q
        self.est['p'][:, 0] = self.Xi.p
        self.est['v'][:, 0] = self.Xi.v
        self.est['ba'][:, 0] = np.zeros(3)
        self.est['bg'][:, 0] = np.zeros(3)
        
        self.std['q'][:, 0] = np.diag(np.sqrt(self.P[0:3, 0:3]))
        self.std['p'][:, 0] = np.diag(np.sqrt(self.P[3:6, 3:6]))
        self.std['v'][:, 0] = np.diag(np.sqrt(self.P[6:9, 6:9]))
        self.std['ba'][:, 0] = np.diag(np.sqrt(self.P[9:12, 9:12]))
        self.std['bg'][:, 0] = np.diag(np.sqrt(self.P[12:15, 12:15]))
        return self
    
    def save_result(self, k):        
        self.est['q'][:, k] = self.Xi.q
        self.est['p'][:, k] = self.Xi.p
        self.est['v'][:, k] = self.Xi.v
        self.est['ba'][:, k] = self.Xi.ba
        self.est['bg'][:, k] = self.Xi.bg
        
        self.std['q'][:, k] = np.sqrt(np.diag(self.P[0:3, 0:3]))
        self.std['p'][:, k] = np.sqrt(np.diag(self.P[3:6, 3:6]))
        self.std['v'][:, k] = np.sqrt(np.diag(self.P[6:9, 6:9]))
        self.std['ba'][:, k] = np.sqrt(np.diag(self.P[9:12, 9:12]))
        self.std['bg'][:, k] = np.sqrt(np.diag(self.P[12:15, 12:15]))
        return self
    
    def propagate(self, IMU0_, dt):
        Xs_idx = range(MSCKF.Dxi, self.P.shape[0])
        
        Pi = self.P[MSCKF.Xi_idx, MSCKF.Xi_idx]
        self.Xi, Pi, F = MSCKF.propagate_imu_euler(self.Xi, Pi, IMU0_, self.Q, dt)
        self.P[MSCKF.Xi_idx, MSCKF.Xi_idx] = Pi
        
        if self.P.shape[0] > MSCKF.Dxi:
            self.P[MSCKF.Xi_idx, Xs_idx] = F @ self.P[MSCKF.Xi_idx, Xs_idx]
            self.P[Xs_idx, MSCKF.Xi_idx] = self.P[MSCKF.Xi_idx, Xs_idx].T              
        return self
    
    def propagate_trapezoidal(self, IMU0_, IMU1_, dt):        
        Xs_idx = slice(MSCKF.Dxi, MSCKF.Dxi + MSCKF.Dxs*len(self.Xs))
        
        Pi = self.P[MSCKF.Xi_idx, MSCKF.Xi_idx]
        self.Xi, Pi, F = MSCKF.propagate_imu_trapezoidal(self.Xi, Pi, IMU0_, IMU1_, self.Q, dt)
        self.P[MSCKF.Xi_idx, MSCKF.Xi_idx] = Pi
        
        if self.P.shape[0] > MSCKF.Dxi:
            Xis = F @ self.P[MSCKF.Xi_idx, Xs_idx]
            self.P[MSCKF.Xi_idx, Xs_idx] = Xis
            self.P[Xs_idx, MSCKF.Xi_idx] = Xis.T
        return self
    
    def correct(self, deadtracks_, IMU0_, dt_):
        if dt_ != 0:
            self.propagate(IMU0_, dt_)
            
        P = MSCKF.validate_cov_matrix(self.P)
        
        Ho, ro, Ro = MSCKF.generate_meas_model(self.Xs, P, self.Rf, self.Rpix, deadtracks_, self.lcam, self.rcam)     
            
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
        
        I_KH = np.eye((K.shape[0], Hz.shape[1])) - K @ Hz        
        P_hat = I_KH @ self.P @ I_KH.T + K @ Rz @ K.T       
        P_hat = MSCKF.validate_cov_matrix(P_hat)
        
        if np.all(np.isreal(np.sqrt(np.diag(P_hat)))):
            del_Xi = del_x_hat[MSCKF.Xi_idx]
            Xs_idx = range(MSCKF.Dxi, len(del_x_hat))
            del_Xs = del_x_hat[MSCKF.Xs]
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
        Xs_aug['q'] = self.Xi.q
        Xs_aug['p'] = self.Xi.p
        Xs_aug['frame'] = frame         
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
        dv0 = (1/2) * (fib_b0 + Rb0b1*fib_b0) * dt_
        dp0 = (1/2) * dv0 * dt_
        gg = Xi_.gg 
        
        # State update
        q = q_gb
        p = Xi_.p + Xi_.v * dt_ + Rgb0 * dp0 + (1/2) * gg * dt_**2 
        v = Xi_.v + Rgb0 * dv0 + gg * dt_
        ba = Xi_.ba 
        bg = Xi_.bg 
        
        Xi = NavState(q, p, v, ba, bg)
        Phi = MSCKF.calc_state_transition_matrix(Xi_, fib_b0, dt_)  # F is phi in here
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
            
        return P_pd
        
    def generate_meas_model(Xs_, P_, Rf_, Rpix_, deadtracks_, lcam_, rcam_):
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
        N_dtracks = len(deadtracks_)
        Ho = np.empty((0, MSCKF.Dxi + MSCKF.Dxs*len(Xs_)))
        ro = np.empty(0)
        Ro = np.empty((0, 0))
        
        for ti in range(N_dtracks):
            track = deadtracks_[ti]
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
                R_j = np.diag(np.tile(np.diag(Rf_), int(len(r_j)/4)))                
                Lj = scipy.linalg.null_space(Hf_j.T)            
                if Lj.shape[0] != 0:
                    Ho_j = Lj.T @ Hx_j
                    ro_j = Lj.T @ r_j
                    Ro_j = Lj.T @ R_j @ Lj
                    
                    gamma = ro_j.T @ np.linalg.inv(Ho_j @ P_ @ Ho_j.T + Ro_j) @ ro_j
                    if gamma < chi2.ppf(MSCKF.chi_trust, len(ro_j)):
                        Ho = np.vstack((Ho, Ho_j))
                        ro = np.hstack((ro, ro_j))
                        Ro = np.block([[Ro, np.zeros((Ro.shape[0], Ro_j.shape[1]))],
                                    [np.zeros((Ro_j.shape[0], Ro.shape[1])), Ro_j]])
                    else:
                        print(f'Track {ti} : Failed to pass Chi-square test !!!')
                        pass                  
                
        return Ho, ro, Ro
                 
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
            
        Tl0ri = Tset[:, :, -1] @ lcam_['Tlr']        # Tl0ri
        two_view_pts = MSCKF.choice_two_points(track_)
        pf_l0_0, bad_status_init  = MSCKF.triangulate_two_view(Tl0ri, lcam_, rcam_, two_view_pts)
        pf_l0, bad_status         = MSCKF.triangulate_multi_stereo_view(track_['un_pts'], Tset, pf_l0_0, lcam_, rcam_, Rpix_)
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
            
            # if z = [X/Z Y/Z]
            # zhat_li = (pf_li[:2]/pf_li[2])[:2]            
            # zhat_ri = (pf_ri[:2]/pf_ri[2])[:2]      
            # zhat = np.hstack((zhat_li.flatten(), zhat_ri.flatten()))
            # zl = np.linalg.solve(lcam_['K'],  np.hstack((track_['un_pts'][:2, i], 1)))
            # zr = np.linalg.solve(rcam_['K'],  np.hstack((track_['un_pts'][2:4, i], 1)))
            # zl = zl[:2]/zl[2]
            # zr = zr[:2]/zr[2]
            # z = np.hstack((zl, zr))
            # residuals[:, i] = z - zhat
            
        deadtrack = track_
        deadtrack['pf_g'] = pf_g        
        deadtrack['residuals'] = residuals        
        deadtrack['bad_status'] = bad_status
        deadtrack['s_idx'] = s_idx                
        # deadtrack['pf_c'] = pf_l0
        # deadtrack['reproj_pts'] = reproj_pts
        return deadtrack
                 
    def triangulate_two_view(Tc0c1_, lcam_, rcam_, pts_):
        """Triangulation with two-view
            Eq : p_c = lambda[0]*v_0 = R01 @ (lambda[1]*v_1) + t01_0
            [v_0, -R01@v_1] @ lambda = t01_0
            Z : depth
            v_0, v_1 : normalized pf_c0, pf_c1
        Args:
            T01_ (_type_): Transformation matrix
            lcam_ (dict): lcam_params
            rcam_ (dict): rcam_params
            pts_ (np.array): 2x2, [u_l, u_r; v_l, v_r]
        Returns:
            _type_: _description_
        """
        Rc0c1 = Tc0c1_[:3, :3]        
        tc0c1 = Tc0c1_[:3, 3]
        pf_c0_n = np.linalg.solve(a=lcam_['K'], b=np.hstack((pts_[:, 0], 1))) # A 가 정방행렬일 때, x = A^(-1) @ b 푸는법
        pf_c1_n = np.linalg.solve(a=rcam_['K'], b=np.hstack((pts_[:, 1], 1))) 
        A = np.hstack((pf_c0_n.reshape(-1, 1), (-Rc0c1 @ pf_c1_n).reshape(-1, 1)))
        b = tc0c1        
        Z = np.linalg.pinv(A) @ b                 # (A^T@A)^-1 @ A^T @ b : Least-square sol.        
        f_c0 = Z[0] * pf_c0_n
        bad_status_init = False
        if Z[0] < 0:
            bad_status_init = True        
        return f_c0, bad_status_init
    
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
                Tril0 = lcam_['Trl'] @ Tlil0             # T^ri_l0
                
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
        
        if (1/x_hat[2] < 0.1) or (1/x_hat[2] > 100) or (C_new[0] > Ntracks*10):
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
        pf_I_l0 = track_['un_pts'][:2, 0]
        pf_I_r0 = track_['un_pts'][2:4, 0]
        
        for i in range(track_['un_pts'].shape[1] - 1):
            if pf_I_l0[0] > track_['un_pts'][0, i+1]:
                pf_I_l0 = track_['un_pts'][:2, i+1]
            elif pf_I_r0[0] < track_['un_pts'][2, i+1]:
                pf_I_r0 = track_['un_pts'][2:4, i+1]
                
        two_view_points = np.hstack((pf_I_l0.reshape(-1, 1), pf_I_r0.reshape(-1, 1)))
        return two_view_points