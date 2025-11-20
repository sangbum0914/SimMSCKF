import numpy as np
import cv2

class Frontend:
    def __init__(self, lcam_, rcam_, Nf_max_, px_dist_):        
        self.lcam = lcam_
        self.rcam = rcam_        
        self.Nf_max = Nf_max_        
        self.px_dist = px_dist_
        self.id_count = 1
        self.view_count = 1
        self.Tracks = []
        self.w_img = 752
        self.h_img = 480
        self.is_init = True
        self.pts0 = np.array([])
        self.pf_g_true = np.empty((5,0))
        
    def set_sim_features(self, pf_g_set_, Tgc_, k, k_vis_):
        Tracks = []
        img0 = np.ones((self.h_img, self.w_img, 3), dtype=np.uint8)*255
        img1 = np.ones((self.h_img, self.w_img, 3), dtype=np.uint8)*255
        
        pf_Il = np.empty((4,0))
        pf_Ir = np.empty((4,0))
        pf_g_true = np.empty((5,0))
        
        for key in pf_g_set_.keys():
            # if key.startswith('pfg_sobj'):                
            #     pf_g_ = pf_g_set_[key]  
            #     img0, img1, pf_Il_, pf_Ir_, pf_g_ = Frontend.project_points(pf_g_, img0, img1, self.lcam, self.rcam, Tgc_,  self.w_img, self.h_img, (0, 255, 0))                
                
            #     pf_Il = np.hstack((pf_Il, pf_Il_))
            #     pf_Ir = np.hstack((pf_Ir, pf_Ir_))    
            #     pf_g_true = np.hstack((pf_g_true, pf_g_))           
            
            # elif key.startswith('pfg_dobj'):  
            #     pf_g_ = pf_g_set_[key][:, :, k]  
            #     img0, img1, pf_Il_, pf_Ir_, pf_g_ = Frontend.project_points(pf_g_, img0, img1, self.lcam, self.rcam, Tgc_,  self.w_img, self.h_img, (0, 0, 255))
                
            #     pf_Il = np.hstack((pf_Il, pf_Il_))
            #     pf_Ir = np.hstack((pf_Ir, pf_Ir_))                           
            #     pf_g_true = np.hstack((pf_g_true, pf_g_))           
                
            # elif key.startswith('pfg_plane'):              
            if key.startswith('pfg_plane'):               
                pf_g_ = pf_g_set_[key]
                img0, img1, pf_Il_, pf_Ir_, pf_g_ = Frontend.project_points(pf_g_, img0, img1, self.lcam, self.rcam, Tgc_,  self.w_img, self.h_img, (255, 0, 0))            
                pf_Il = np.hstack((pf_Il, pf_Il_))
                pf_Ir = np.hstack((pf_Ir, pf_Ir_))                       
                pf_g_true = np.hstack((pf_g_true, pf_g_))   
                
        pf_Il_inlier = pf_Il
        pf_Ir_inlier = pf_Ir
        
        Nf = pf_Il_inlier.shape[1]
        for j in range(Nf):
            track = {'feature_id'  : int(pf_Il_inlier[0, j]),
                    'view_id'      : k_vis_,
                    'uv'           : np.hstack((pf_Il_inlier[1:3, j], pf_Ir_inlier[1:3, j])),          # [u_l, v_l, u_r, v_r]
                    'un_uv'        : np.hstack((pf_Il_inlier[1:3, j], pf_Ir_inlier[1:3,j])),
                    'object_id'       : pf_Il_inlier[3, j]}
            Tracks.append(track)     
                
        self.Tracks = Tracks
        self.is_zupt = False
        self.pf_g_true = pf_g_true
        img0 = cv2.flip(img0, 0)
        img1 = cv2.flip(img1, 0)
        self.img0 = cv2.flip(img0, 1)
        self.img1 = cv2.flip(img1, 1)
        return self
        
    @ staticmethod
    def project_points(pf_g_, img0_, img1_, lcam_, rcam_, Tgc_,  w_img_, h_img_, color_):        
        pf_l = np.linalg.solve(Tgc_, np.vstack([pf_g_[1:4, :], np.ones((1, pf_g_.shape[1]))]))  # [X, Y, Z, 1]^l
        pf_l_n = np.vstack([pf_l[0, :] / pf_l[2, :],
                            pf_l[1, :] / pf_l[2, :],
                            pf_l[2, :] / pf_l[2, :]])
        pf_Il = lcam_['K'] @ pf_l_n
        pf_Il = np.vstack((pf_g_[0, :], pf_Il))
        
        pf_r = np.linalg.solve(lcam_['Tlr'], pf_l)
        pf_r_n = np.vstack([pf_r[0, :] / pf_r[2, :],
                            pf_r[1, :] / pf_r[2, :],
                            pf_r[2, :] / pf_r[2, :]])
        pf_Ir = rcam_['K'] @ pf_r_n                
        pf_Ir = np.vstack((pf_g_[0, :], pf_Ir))
        
        val_idx = (pf_l[2, :] > 0) & (pf_r[2, :] > 0) & \
                (pf_Il[1, :] > 1) & (pf_Il[1, :] < w_img_) & \
                (pf_Il[2, :] > 1) & (pf_Il[2, :] < h_img_) & \
                (pf_Ir[1, :] > 1) & (pf_Ir[1, :] < w_img_) & \
                (pf_Ir[2, :] > 1) & (pf_Ir[2, :] < h_img_)

        pf_Il = np.vstack((pf_Il[:3, :], pf_g_[4, :]))
        pf_Ir = np.vstack((pf_Ir[:3, :], pf_g_[4, :]))
        
        pf_Il = pf_Il[:, val_idx] 
        pf_Ir = pf_Ir[:, val_idx]         
        pf_g = pf_g_[:, val_idx]
        
        # Noise addition : N~(0,0.3^2)        
        pf_Il[1:3, :] = pf_Il[1:3, :] + 0.3 * np.random.randn(2, pf_Il.shape[1])
        pf_Ir[1:3, :] = pf_Ir[1:3, :] + 0.3 * np.random.randn(2, pf_Il.shape[1])
        
        pf_Il_int = np.round(pf_Il).astype(np.int32)
        pf_Ir_int = np.round(pf_Ir).astype(np.int32)
        
        for u, v in pf_Il_int[1:3].T:
            cv2.circle(img0_, (u, v), radius=2, color=color_, thickness=-1)
            
        for u, v in pf_Ir_int[1:3].T:
            cv2.circle(img1_, (u, v), radius=2, color=color_, thickness=-1)
            
        return img0_, img1_, pf_Il, pf_Ir, pf_g

                        
                
                        
    
        