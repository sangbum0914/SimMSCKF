import numpy as np
from msckf import MSCKF
from utils.attitude import Attitude
from utils.SO3 import SO3

import matplotlib.pyplot as plt

class ObjectProcessor:
    def __init__(self, min_slw_, max_slw_, lcam_, rcam_):
        self.min_slw = min_slw_
        self.max_slw = max_slw_
        self.lcam = lcam_
        self.rcam = rcam_
        
        self.pTgl = np.zeros((4, 4))
        self.c_ts = 0
        self.p_ts = 0
        
        self.is_init = True
        self.object_map = {}        # {object_id : {'dyna_state' : Bool, view_id : [fid; uv, un_uv, fl, fg]}                                    
        self.dead_objects = []                                          
        self.dS_bar = {}
        self.static_deadtracks = []
        self.dynamic_deadtracks = []
        
        self.thres_dyna = 3
        # self.thres_dyna = 99999
        
    # def detect_dynamic_and_mapping(self, tracks_, cur_view_id_, Xmsckf_, c_ts_, fg_true):   
    def detect_dynamic_and_mapping(self, tracks_, cur_view_id_, Xmsckf_, c_ts_):          
        N_tracks = len(tracks_)
        if N_tracks == 0:
            # print("Object track is empty !!!")
            return self  
        
        cTgb = np.block([[Attitude.quat2dcm(Xmsckf_.Xi.q), Xmsckf_.Xi.p.reshape(-1, 1)],
                        [0, 0, 0, 1]])
        cTgl = cTgb @ np.linalg.inv(self.lcam['Tcb'])    
        
        # Stereo view triangulation
        un_uv = np.zeros((4, N_tracks))     # undistorted points, 4xNf
        for j in range(N_tracks):            
            un_uv[:, j] = tracks_[j]['un_uv']      
        fl, val_idx = MSCKF.triangulate_stereo_view(lcam_=Xmsckf_.lcam, rcam_=Xmsckf_.rcam, un_pts_=un_uv) 
        # Filtering tracks for valid stereo view depth, (Z < 0).
        vtracks = [tracks_[j] for j in val_idx]
        tracks_ = vtracks                    
        fg = cTgl[:3, :3] @ fl + cTgl[:3, 3].reshape(-1, 1)
        
        N_tracks = len(tracks_)        
        if N_tracks == 0:
            print("All triangulation has failed !!!")            
            return   
        
        # Inserting to object map
        for j in range(N_tracks):
            object_id = tracks_[j]['object_id']                       
            view_id = tracks_[j]['view_id']
            value = np.hstack((tracks_[j]['feature_id'], tracks_[j]['uv'], tracks_[j]['un_uv'], fl[:,j], fg[:,j])).reshape(-1, 1)                
            if object_id not in self.object_map:
                self.object_map[object_id] = {'dyna_state' : []}
                self.object_map[object_id][view_id] = value
            else:           
                if view_id not in self.object_map[object_id]:
                    self.object_map[object_id][view_id] = value
                else:
                    self.object_map[object_id][view_id] = np.hstack((self.object_map[object_id][view_id], value))             
        
        # Remove objects if the Nf is under 10
        remove_object_ids = []
        for object_id, object_values in self.object_map.items():
            Nf_object = object_values[cur_view_id_].shape[1]        # Number of tracked features in objects
            if Nf_object < 10:
                remove_object_ids.append(object_id)
        
        if len(remove_object_ids) > 0:
            for remove_object_id in remove_object_ids:
                self.object_map.pop(remove_object_id)
        
        if self.is_init:                                
            self.pTgl = cTgl
            self.pTs = c_ts_
            self.is_init = False   
            return self                 
        else:                      
            # Filtering out the untracked features
            for object_id, object_values in self.object_map.items():
                view_ids = [key for key in object_values.keys() if isinstance(key, (int, float))]
                if len(view_ids) < 2:
                    continue
                    
                feature_id_km1 = object_values[cur_view_id_-1][0, :]           # k_last - 1
                feature_id_k = object_values[cur_view_id_][0, :]               # k_last 
                _, idx_km1, idx_k = np.intersect1d(feature_id_km1, feature_id_k, return_indices=True)
                
                for view_ids in [key for key in object_values.keys() if isinstance(key, (int, float))][:-1]:   # Treat past tracks & cur_view_tracks differently
                    self.object_map[object_id][view_ids] = object_values[view_ids][:, idx_km1]                 # Filter all the past tracks w.r.t the km1 tracks                 
                self.object_map[object_id][cur_view_id_] = object_values[cur_view_id_][:, idx_k]
                
            # Dynamic state detector
            for object_id, object_values in self.object_map.items():
                view_ids = [key for key in object_values.keys() if isinstance(key, (int, float))]
                if len(view_ids) < 2:
                    continue
                dyna_state, dS_mdist, dS_mdist_bar = ObjectProcessor.detect_dynamic_state(object_values, cur_view_id_, cTgl, self.pTgl, (c_ts_ - self.p_ts), self.lcam, self.rcam, self.thres_dyna)
                
                # To check average mahalanobis distance of the scene flow rate
                if object_id not in self.dS_bar:
                    self.dS_bar[object_id] = np.hstack((c_ts_, dS_mdist_bar))
                else:
                    self.dS_bar[object_id] = np.vstack((self.dS_bar[object_id], np.hstack((c_ts_, dS_mdist_bar))))  
                    
                if cur_view_id_ == 1100:
                    stop = 1
                    
                if dyna_state:
                    self.object_map[object_id]['dyna_state'].append(True)
                else:
                    self.object_map[object_id]['dyna_state'].append(False)
        
        # Detect dead objects         
        remove_object_ids = []
        dead_objects_candidates = []        
        for object_id, object_values in self.object_map.items():
            view_ids = [key for key in object_values.keys() if isinstance(key, (int, float))]  
            # List up the dead object candidates and remove_ids
            if view_ids[-1] != cur_view_id_:
                remove_object_ids.append(object_id)
                dead_objects_candidates.append(self.object_map[object_id])            
            elif len(view_ids) > self.max_slw:
                object_values_save = {k: object_values[k] for k in view_ids[:-1]}
                object_values_save['dyna_state'] = ObjectProcessor.calc_majority_state(object_values['dyna_state'])
                remove_object_ids.append(object_id)
                dead_objects_candidates.append({object_id : object_values_save})
            elif object_id in {10, 11, 12, 13, 14}:
                remove_object_ids.append(object_id)
                
        # Remove from object map 
        for remove_key in remove_object_ids:            
            self.object_map.pop(remove_key, None)            
            
        # Return dead objects
        self.dead_objects = dead_objects_candidates                 
        self.pTgl = cTgl
        self.p_ts = c_ts_
        return self
     
    def detect_dynamic_state(object_values_, cur_view_id_, cTgl_, pTgl_, dt_, lcam_, rcam_, thres_dyna_):
        fl_km1 = object_values_[cur_view_id_-1][9:12,:]
        fl_k = object_values_[cur_view_id_][9:12,:]
        T_lk_lkm1 = np.linalg.inv(cTgl_) @ pTgl_
        dS_maha_dist = ObjectProcessor.calc_scene_flow_rate_mahalanobis_dist(fl_k, fl_km1, T_lk_lkm1, dt_, lcam_, rcam_)
        # dS_bar = np.mean(dS_maha_dist)      
        dS_bar = np.median(dS_maha_dist)      
        
        dyna_state = False
        if dS_bar > thres_dyna_:
            dyna_state = True
            
        return dyna_state, dS_maha_dist, dS_bar
     
    def calc_scene_flow_rate_mahalanobis_dist(fl_k_, fl_km1_, T_lk_lkm1_, dt_, lcam_, rcam_):       
        # Scene flow rate
        dS = (fl_k_ - (T_lk_lkm1_[:3,:3] @ fl_km1_ + T_lk_lkm1_[:3,3].reshape(-1, 1)))/dt_     
        # Reset flk, flkm1, dS with stochastic approach : (2 sigma from median value)
        dS_dist = np.diag(dS.T @ dS)                      
        med_dS_dist = np.median(dS_dist)
        sig_dS_dist = np.std(dS_dist)
        z_scores = (dS_dist - med_dS_dist) / sig_dS_dist                        
        fl_km1_ = fl_km1_[:, np.abs(z_scores) < 2] 
        fl_k_ = fl_k_[:, np.abs(z_scores) < 2]
        dS = dS[:, np.abs(z_scores) < 2]                 
        
        # Calc Sigma_dM
        Cov_dM = np.zeros((8, 8))
        sig_pixel = 3**2
        sig_dtheta = (0.5*np.pi/180)**2  
        sig_dp = 0.0001**2             
        Cov_dM[0:2, 0:2] = np.diag([sig_pixel, sig_pixel])                                  # 1     [pix]         # Cov_dM[0:2, 0:2] = np.diag([Xmsckf_.Rpix, Xmsckf_.Rpix])
        Cov_dM[2:5, 2:5] = np.diag([sig_dtheta, sig_dtheta, sig_dtheta])                    # 1e-3  [deg]         # Xmsckf_.P[Xmsckf_.q_idx,Xmsckf_.q_idx]
        Cov_dM[5:, 5:] = np.diag([sig_dp, sig_dp, sig_dp])                                  # 0.001 [m]           # Xmsckf_.P[Xmsckf_.p_idx,Xmsckf_.p_idx]     
        
        
        dS_maha_dist = np.zeros(fl_km1_.shape[1])
        for j in range(fl_km1_.shape[1]):
            # # Normalized SFR Method : # df_du = np.vstack((np.diag([1/Xmsckf_.lcam['K'][0,0], 1/Xmsckf_.lcam['K'][1,1]]), np.array([0, 0]).reshape(1,-1))) # df_dtheta = SO3.skew(T_lk_lkm1[:3,:3] @ nfl_km1[:, j])
            # Scene flow rate method
            df_du = np.vstack((np.diag([fl_km1_[2, j]/lcam_['K'][0,0], fl_km1_[2, j]/lcam_['K'][1,1]]), np.array([0, 0]).reshape(1,-1)))            
            df_dtheta = SO3.skew(T_lk_lkm1_[:3,:3] @ fl_km1_[:, j])            
            df_dp = np.diag([1, 1, 1])
            JM = np.hstack((df_du, df_dtheta, df_dp))
            Cov_dS = JM @ Cov_dM @ JM.T
            dS_maha_dist[j] = np.sqrt(dS[:,j].T @ np.linalg.inv(Cov_dS) @ dS[:,j])  
            
        return dS_maha_dist
    
    def calc_majority_state(boolean_list_):
        true_count = sum(boolean_list_)
        false_count = len(boolean_list_) - true_count
        return True if true_count >= false_count else False 