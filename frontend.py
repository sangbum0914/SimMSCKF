import numpy as np
import cv2
from scipy.spatial import cKDTree

class Frontend:
    def __init__(self, configs_):        
        self.lcam = configs_['lcam']
        self.rcam = configs_['lcam']
        self.Nf_max = configs_['max_cnt']
        self.px_dist = configs_['min_dist']
        self.F_threshold = configs_['F_threshold']
        self.static_threshold = configs_['static_threshold']
        self.w_img = configs_['image_width']
        self.h_img = configs_['image_height']
        
        self.id_count = 1
        self.view_count = 1
        self.sTracks = []
        self.oTracks = {}
                        
        self.is_init = True             
        self.is_zupt = False
        
        self.fimg0 = []
        self.simg0 = []
        self.pimg0 = []
        self.pimg1 = []
        self.ppts0 = np.empty((0,2), dtype='float32')
        self.ppts1 = np.empty((0,2), dtype='float32')               
            
        # VIODE dataset
        # self.rgb_to_id = {
        #             (115,237,170) : 0,
        #             (255,169,239) : 1,
        #             (99,202,227)  : 2,
        #             (236,239,160) : 3,
        #             (143,234,243) : 4,
        #             (166,221,155) : 5,
        #             (154,248,245) : 6,
        #             (253,210,188) : 7,
        #             (226,59,251)  : 8,
        #             (108,91,207)  : 9,
        #             (243,196,231) : 10,
        #             (209,231,181) : 11,
        #             (114,119,232) : 12} 
        
        self.rgb_to_id = {(0, 0, 0) : 0}
        
        self.unusable_id = {
                    (164,241,109) : 140,     # tree
                    (207,131,215) : 151,     # vegetation
                    (214,214,194) : 207}     # sky
                    
        
    def set_images(self, img0_, img1_, simg0_, ts_, k_vis_):
        self.view_count = k_vis_
        self.fimg0 = cv2.cvtColor(img0_, cv2.COLOR_BGR2RGB)
        self.simg0 = cv2.cvtColor(img0_, cv2.COLOR_BGR2RGB)
        
        # Track points if past feas exists
        if (self.ppts0.shape[0] != 0):            
            ppts0, cpts0, un_ppts0, un_cpts0, val_idx0 = Frontend.two_view_tracking(self.ppts0, self.pimg0, img0_, self.lcam, self.rcam, self.F_threshold)
            # ZUPT analysis
            if (un_cpts0[val_idx0, :].shape[0] > 20) and (np.mean(np.linalg.norm((un_cpts0[val_idx0, :] - un_ppts0[val_idx0, :]), axis=1)) < self.static_threshold):
                self.is_zupt = True
                return self
            else:
                self.is_zupt = False
                # Static tracking
                cpts0, cpts1, un_cpts0, un_cpts1, val_idx1 = Frontend.two_view_tracking(cpts0, img0_, img1_, self.lcam, self.rcam, self.F_threshold)
                val_idx = val_idx0 & val_idx1
                cpts0 = cpts0[val_idx, :]
                cpts1 = cpts1[val_idx, :]
                un_cpts0 = un_cpts0[val_idx, :]
                un_cpts1 = un_cpts1[val_idx, :]
                self.fimg0 = Frontend.draw_points_on_image(self.fimg0, cpts0)
                
                tracked_id_ = np.where(val_idx)[0]
                Tracks = []
                for j in range(cpts0.shape[0]):
                    track = {'feature_id'  : self.sTracks[tracked_id_[j]]['feature_id'],
                            'view_id'      : self.view_count,
                            'uv'           : np.hstack((cpts0[j], cpts1[j])),          # [u_l, v_l, u_r, v_r]
                            'un_uv'        : np.hstack((un_cpts0[j], un_cpts1[j])),
                            'object_id'    : -1}
                    Tracks.append(track)
                self.sTracks = Tracks
                self.ppts0 = cpts0
                self.ppts1 = cpts1

        # Extract new features
        if (self.ppts0.shape[0] < self.Nf_max):
            # Static Instances
            static_mask, roi_img0 = Frontend.mask_static_objects(img0_, simg0_, self.rgb_to_id)   
            npts0 = Frontend.extract_points(img0_, static_mask)                     
            npts0 = Frontend.select_uniform_points(self.ppts0, npts0, self.Nf_max - self.ppts0.shape[0], self.px_dist)            
            if (npts0.shape[0] == 0):
                print("Extraction of new features is failed...")
                # raise ValueError("Extraction of new features is failed...")
            else:
                npts0, npts1, un_npts0, un_npts1, val_idx = Frontend.two_view_tracking(npts0, img0_, img1_, self.lcam, self.rcam, self.F_threshold)
                npts0 = npts0[val_idx, :]
                npts1 = npts1[val_idx, :]
                un_npts0 = un_npts0[val_idx, :]
                un_npts1 = un_npts1[val_idx, :]
                self.fimg0 = Frontend.draw_points_on_image(self.fimg0, npts0)
                        
                new_Tracks = []
                for j in range(npts0.shape[0]): 
                    track = {'feature_id'   : self.id_count,
                            'view_id'      : self.view_count,
                            'uv'           : np.hstack((npts0[j], npts1[j])),          # [u_l, v_l, u_r, v_r]
                            'un_uv'        : np.hstack((un_npts0[j], un_npts1[j])),
                            'object_id'    : -1}
                    new_Tracks.append(track)
                    self.id_count += 1
                """
                for rgb, object_id in self.rgb_to_id.items():
                    id_count_opts = 1
                    object_mask = np.all(simg0_ == rgb[::-1], axis=-1).astype(np.uint8) * 255                
                    if np.any(object_mask):
                        opts0 = Frontend.extract_points(img0_, object_mask)
                        opts0, opts1, un_opts0, un_opts1, val_idx = Frontend.two_view_tracking(opts0, img0_, img1_, self.lcam, self.rcam, self.F_threshold)
                        opts0 = opts0[val_idx, :] 
                        opts1 = opts1[val_idx, :]
                        un_opts0 = un_opts0[val_idx, :]
                        un_opts1 = un_opts1[val_idx, :]    
                        self.simg0 = Frontend.overlay_mask_on_image(img0_=img0_, mask_=object_mask, mask_color_=rgb[::-1])
                        
                        self.oTracks[object_id]
                        ofeas = []                    
                        for j in range(opts0.shape[0]):
                            track.append(np.array([self.id_count, pts0[j], pts1[j], un_pts0[j], un_pts1[j]]))
                            ofeas.append(track)
                            id_count_opts += 1
                        self.oTracks[object_id][self.view_count] = np.array(ofeas)
                """                   
                self.sTracks = self.sTracks + new_Tracks
                self.ppts0 = np.vstack((self.ppts0, npts0))
                self.ppts1 = np.vstack((self.ppts1, npts1))
            # Movable Instances
            
            
        self.pimg0 = img0_
        self.pimg1 = img1_
        return self


        if (self.is_init) or (self.pts0.size == 0):
            static_mask, roi_img0 = Frontend.mask_static_objects(img0_, simg0_, self.rgb_to_id)            
            # Static-instance tracking
            pts0 = Frontend.extract_points(img0_, static_mask)
            pts0 = Frontend.select_uniform_points([], pts0, self.Nf_max, self.px_dist)         
            pts0, pts1, un_pts0, un_pts1, val_idx = Frontend.two_view_tracking(pts0, img0_, img1_, self.lcam, self.rcam, self.F_threshold)            
            pts0 = pts0[val_idx, :]
            pts1 = pts1[val_idx, :]
            un_pts0 = un_pts0[val_idx, :]
            un_pts1 = un_pts1[val_idx, :]
            fimg0 = Frontend.draw_points_on_image(cv2.cvtColor(img0_, cv2.COLOR_BGR2RGB), pts0)
            # cv2.imshow('fimg0', fimg0)
            # cv2.waitKey(0)            
            for j in range(pts0.shape[0]):
                track = {'feature_id'   : self.id_count,
                         'view_id'      : self.view_count,
                         'uv'           : np.hstack((pts0[j], pts1[j])),          # [u_l, v_l, u_r, v_r]
                         'un_uv'        : np.hstack((un_pts0[j], un_pts1[j])),
                         'object_id'    : -1}
                self.sTracks.append(track)
                self.id_count += 1
                
            # Movable-instance tracking     
            simg0 = cv2.cvtColor(img0_, cv2.COLOR_BGR2RGB)           
            for rgb, object_id in self.rgb_to_id.items():
                id_count_opts = 1
                object_mask = np.all(simg0_ == rgb[::-1], axis=-1).astype(np.uint8) * 255                
                
                if np.any(object_mask):
                    opts0 = Frontend.extract_points(img0_, object_mask)
                    opts0, opts1, un_opts0, un_opts1, val_idx = Frontend.two_view_tracking(pts0, img0_, img1_, self.lcam, self.rcam, self.F_threshold)
                    opts0 = opts0[val_idx, :] 
                    opts1 = opts1[val_idx, :]
                    un_opts0 = un_opts0[val_idx, :]
                    un_opts1 = un_opts1[val_idx, :]    
                    simg0 = Frontend.overlay_mask_on_image(img0_=img0_, mask_=object_mask, mask_color_=rgb[::-1])
                    
                    self.oTracks[object_id]
                    ofeas = []                    
                    for j in range(opts0.shape[0]):
                        track.append(np.array([self.id_count, pts0[j], pts1[j], un_pts0[j], un_pts1[j]]))
                        ofeas.append(track)
                        id_count_opts += 1
                    self.oTracks[object_id][self.view_count] = np.array(ofeas)
            
            self.is_init = False
            self.is_zupt = False
            self.fimg0 = fimg0
            self.simg0 = simg0
            self.pimg0 = img0_
            self.pimg1 = img1_
            self.pts0 = pts0
            self.pts1 = pts1           
        else:
            # Temporal tracking
            ppts0 = self.pts0
            ppts0, cpts0, _, _, val_idx0 = Frontend.two_view_tracking(ppts0, self.pimg0, img1_, self.lcam, self.lcam, self.F_threshold)            
            ppts0_tmp = ppts0[val_idx0, :]  # temporary space for ZUPT analysis
            cpts0_tmp = cpts0[val_idx0, :]
            # is_zupt = Frontend.check_static(img0_, img1_, self.lcam, self.rcam, self.F_threshold, self.static_threshold)
            
            if (cpts0_tmp.shape[0] > 20) and (np.mean(np.linalg.norm((ppts0_tmp - cpts0_tmp))) < self.static_threshold):
                self.is_zupt = True
                return 
            else:
                self.is_zupt = False                
                static_mask, roi_img0 = Frontend.mask_static_objects(img0_, simg0_, self.rgb_to_id)
                # Static tracking
                cpts0, cpts1, un_cpts0, un_cpts1, val_idx1 = Frontend.two_view_tracking(cpts0, img0_, img1_, self.lcam, self.rcam, self.F_threshold)            
                val_idx = val_idx0 & val_idx1
                cpts0 = cpts0[val_idx, :]
                cpts1 = cpts1[val_idx, :]
                un_cpts0 = un_cpts0[val_idx, :]
                un_cpts1 = un_cpts1[val_idx, :]
                
                Nf = cpts0.shape[0]
                tracked_id_ = np.where(val_idx)[0]
                Tracks0 = []
                for j in range(Nf):
                    track = {'feature_id'  : self.Tracks[tracked_id_[j]]['feature_id'],
                            'view_id'      : self.view_count,
                            'uv'           : np.hstack((cpts0[j], cpts1[j])),          # [u_l, v_l, u_r, v_r]
                            'un_uv'        : np.hstack((un_cpts0[j], un_cpts1[j])),
                            'object_id'    : -1}
                    Tracks0.append(track)
                    
                # New features
                Tracks1 = []
                npts0 = np.array([])
                npts1 = np.array([])
                if Nf < self.Nf_max:
                    pts_new = Frontend.extract_points(img0_, static_mask)
                    npts0 = Frontend.select_uniform_points(cpts0, pts_new, self.Nf_max - Nf, self.px_dist)
                    
                    if (npts0.shape[0] != 0):
                        npts0, npts1, un_npts0, un_npts1, val_idx = Frontend.two_view_tracking(npts0, img0_, img1_, self.lcam, self.rcam, self.F_threshold)
                        npts0 = npts0[val_idx, :]
                        npts1 = npts1[val_idx, :]
                        un_npts0 = un_npts0[val_idx, :]
                        un_npts1 = un_npts1[val_idx, :]
                    
                    Nf = npts0.shape[0]
                    for j in range(Nf):
                        track = {'feature_id'   : self.id_count,
                                'view_id'      : self.view_count,
                                'uv'           : np.hstack((npts0[j], npts1[j])),          # [u_l, v_l, u_r, v_r]
                                'un_uv'        : np.hstack((un_npts0[j], un_npts1[j])),
                                'object_id'    : -1}
                        Tracks1.append(track)
                        self.id_count += 1
                        
                self.Tracks = Tracks0 + Tracks1
                if npts0.shape[0] != 0:
                    self.pts0 = np.vstack((cpts0, npts0))
                    self.pts1 = np.vstack((cpts1, npts1)) 
                else:
                    self.pts0 = cpts0
                    self.pts1 = cpts1                   
                self.pimg0 = img0_
                self.pimg1 = img1_
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
        px_noise = 0.1 * np.random.randn(2, pf_Il.shape[1])
        pf_Il[1:3, :] = pf_Il[1:3, :] + px_noise
        pf_Ir[1:3, :] = pf_Ir[1:3, :] + px_noise
        
        pf_Il_int = np.round(pf_Il).astype(np.int32)
        pf_Ir_int = np.round(pf_Ir).astype(np.int32)
        
        for u, v in pf_Il_int[1:3].T:
            cv2.circle(img0_, (u, v), radius=2, color=color_, thickness=-1)
            
        for u, v in pf_Ir_int[1:3].T:
            cv2.circle(img1_, (u, v), radius=2, color=color_, thickness=-1)
            
        return img0_, img1_, pf_Il, pf_Ir, pf_g
    
    def extract_points(img0_, mask_, options_='GFTT'):
        """Extract feature points from an input image 
        Args:
            img0_ (cv::Mat): input image
        Returns:
            pts0 (np.array): Nx2, uv, location of extracted points at image plane
        """
        if options_ == 'GFTT':
            if len(img0_.shape) == 3:
                img0_ = cv2.cvtColor(img0_, cv2.COLOR_BGR2GRAY)
            pts0 = cv2.goodFeaturesToTrack(img0_, maxCorners=9999, qualityLevel=0.01, minDistance=7, mask=mask_, blockSize=7, useHarrisDetector=False, k=0.04)
            pts0 = pts0.reshape(-1,2).astype(np.float32)
        elif options_ == 'ORB':        
            orb = cv2.ORB_create(nfeatures=99999)
            kpts0, des0 = orb.detectAndCompute(img0_, mask_)
            skpts0 = sorted(kpts0, key=lambda kp: kp.response, reverse=True)        
            skpts0 = [kp for kp in skpts0 if kp.size > 30]
            pts0 = np.array([kp.pt for kp in skpts0], dtype=np.float32)
        elif options_== 'FAST':
            if len(img0_.shape) == 3:
                img0_ = cv2.cvtColor(img0_, cv2.COLOR_BGR2GRAY)
            fast = cv2.FastFeatureDetector_create(threshold=15, nonmaxSuppression=True)
            kpts0 = fast.detect(img0_, mask_)
            pts0 = np.array([kp.pt for kp in kpts0], dtype=np.float32)
        return pts0    
    
    def mask_static_objects(img0_, simg0_, rgb_to_id):
        mask = np.zeros((img0_.shape[0], img0_.shape[1]), dtype=np.uint8)
        for rgb, class_id in rgb_to_id.items():
            mask |= np.all(simg0_ == rgb, axis=-1).astype(np.uint8) * 255            
        mask = cv2.bitwise_not(mask)
        roi_image = cv2.bitwise_and(simg0_, simg0_, mask=mask)                  
        return mask, roi_image
    
    def draw_points_on_image(img0_, pts0_):
        fimg0 = img0_.copy()
        for point in pts0_:
            x, y = int(point[0]), int(point[1])
            cv2.circle(fimg0, (x, y), radius=3, color=(0, 255, 0), thickness=-1)        
        return fimg0
    
    def overlay_mask_on_image(img0_, mask_, mask_color_):
        simg0 = img0_.copy()
        mask_colored = np.zeros_like(img0_, dtype=np.uint8)
        mask_colored[mask_ > 0] = mask_color_
        cv2.addWeighted(mask_colored, 0.5, simg0, 0.5, 0, dst=simg0)
        return simg0
    
    def select_uniform_points(ppts0_, npts0_, Nf_needed_, px_dist_):
        """Select uniform points with previously existed points and newly extracted points
        Args:
            ppts0_ (np.array): Nx2, previous points
            npts0_ (np.array): Nx2, new points 
            Nf_needed_ (int): number of features to newly extract
            px_dist_ (int): minimum pixel distance between features
        Returns:
            npts0_selected (np.array) : selected new points
        """   
        
        npts0_selected = np.empty((0, 2), dtype='float32')
        if ppts0_.shape[0] == 0:
            tree = None
            for tmp_pt in npts0_:
                if tree == None:
                    tree = cKDTree(tmp_pt.reshape(1, -1))  # Initial build of KDTree
                    continue
                min_dist, _ = tree.query(tmp_pt)
                if min_dist > px_dist_:
                    npts0_selected = np.vstack((npts0_selected, tmp_pt))
                    tree = cKDTree(npts0_selected)
                if npts0_selected.shape[0] >= Nf_needed_:
                    break    
        else:
            tree = cKDTree(ppts0_)            
            for tmp_pt in npts0_:
                min_dist, _ = tree.query(tmp_pt)
                if min_dist > px_dist_:
                    npts0_selected = np.vstack((npts0_selected, tmp_pt))
                    tree = cKDTree(np.vstack((ppts0_, npts0_selected)))
                if npts0_selected.shape[0] >= Nf_needed_:
                    break    
        return npts0_selected    

    def two_view_tracking(pts0_, img0_, img1_, lcam_, rcam_, F_threshold_):
        klt_params = dict(winSize=(21, 21), maxLevel=15, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
        pts1, st, err = cv2.calcOpticalFlowPyrLK(img0_, img1_, pts0_, None, **klt_params)  
        
        # Reverse check img1 -> img0
        pts0_reproj, st_back, err_back = cv2.calcOpticalFlowPyrLK(
            img1_, img0_, pts1, pts0_.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **klt_params
        )
        
        fb_dist = np.linalg.norm(pts0_ - pts0_reproj, axis=1)
        fb_thresh = 0.5
        fb_consistent = fb_dist < fb_thresh
        
        
        un_pts0 = Frontend.undistort_points(pts0_, lcam_)
        un_pts1 = Frontend.undistort_points(pts1,  rcam_)
        
        # RANSAC using F matrix
        try:
            _, mask = cv2.findFundamentalMat(un_pts0, un_pts1, cv2.FM_RANSAC, ransacReprojThreshold=F_threshold_, confidence=0.999, maxIters=500)
            ransac_idx = mask.ravel() == 1
        except:
            ransac_idx = np.zeros((un_pts0.shape[0]), dtype=bool)
            
        h, w  = img0_.shape[:2]
        val_idx = (st.flatten() == 1) & ransac_idx & fb_consistent &\
                (pts1[:, 0] > 0) & (pts1[:, 0] < w) & (pts1[:, 1] > 0) & (pts1[:, 1] < h)                
        return pts0_, pts1, un_pts0, un_pts1, val_idx
                
    def undistort_points(pts0_, cam_):
        """Undistort feature points with radtan(pinhole) model
        Args:
            pts0_ (np.array): Nx2, uv, feature points at image plane
            cam_ (dict): camera parameters, K, dc
        Returns:
            upts0 (np.array) : Nx2, undistorted feature points
        """
        if cam_['dc'][0] == 0:  # No distortion 
            upts0 = pts0_        
        else:            
            pf_I = np.vstack(pts0_.T, np.ones((1, pts0_.shape[0])))    # pf_I = [u, v, 1], 3xN
            pf_c_n = np.linalg.solve(cam_['K'], pf_I)                               # pf_c_n = [X/Z, Y/Z, 1], 3xN
            x_n = pf_c_n[0, :]
            y_n = pf_c_n[1, :]        
            max_iter = 20
            
            if cam_['dist_model'] == 'PINHOLE':
                r2 = x_n**2 + y_n**2
                for i in range(max_iter):                    
                    ic_dist = 1 / (1 + ((0 * r2 + cam_['dc'][1]) * r2 + cam_['dc'][0]) * r2)    # icdist : inverse coefficients of distortion
                    dx = 2 * cam_['dc'][2] * x_n * y_n + cam_['dc'][3] * (r2 + 2 * y_n**2)
                    dy = cam_['dc'][2] * (r2 + 2 * y_n**2) + 2*cam_['dc'][3] * x_n * y_n
                    x_n_hat = (x_n - dx) * ic_dist
                    y_n_hat = (y_n - dy) * ic_dist
                    
            elif cam_['dist_model'] == 'equidistant':                               # Schneith's implementation : iterative solver of inverse equidistant model
                r = np.sqrt(x_n**2 + y_n**2)
                for i in range(max_iter):
                    r2 = r**2
                    r4 = r2**2
                    r6 = r2*r4
                    r8 = r4*r4
                    r_hat = r / (1 + cam_['dc'][0] * r2 + cam_['dc'][1] * r4 + cam_['dc'][2] * r6 + cam_['dc'][3] * r8)                    
                s = np.tan(r_hat) / r   # scaling
                x_n_hat = s * x_n
                y_n_hat = s * y_n      
                                             
            upts0 = cam_['K'] @ np.vstack(x_n_hat, y_n_hat, np.ones(1, x_n_hat.shape[0]))   # [u, v, 1] : 3xN
            upts0 = upts0.T[:, 2]                                                           # [u, v] : Nx2 
        return upts0
        
    def check_static(img0_, img1_, lcam_, rcam_, F_threshold_, static_threshold_):      
        pts0 = Frontend.extract_points(img0_, None)
        _, _, un_pts0, un_pts1, _ = Frontend.two_view_tracking(pts0, img0_, img1_, lcam_, rcam_, F_threshold_)
        dpts = un_pts0 - un_pts1
        if np.mean(np.linalg.norm(dpts, axis=1)) < static_threshold_:
            is_vehicle_static  = True
        else:
            is_vehicle_static  = False                                
        return is_vehicle_static
        
        
                        
                
                        
    
        