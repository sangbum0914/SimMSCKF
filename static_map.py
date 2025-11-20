import numpy as np

class StaticMap:
    def __init__(self, min_slw_, max_slw_):
        self.min_slw = min_slw_
        self.max_slw = max_slw_
        self.map_server = {}        # key   : feature_id  
                                    # value : view_id, uv, un_uv
        self.dead_tracks = []
        
    def add_features(self, tracks_, cur_view_id_):
        N_tracks = len(tracks_)
        if N_tracks == 0:
            print("track is empty !!!")
            return
        for j in range(N_tracks):            
            key = tracks_[j]['feature_id']           
            value = np.hstack((tracks_[j]['view_id'], tracks_[j]['uv'], tracks_[j]['un_uv'])).reshape(-1, 1)
            if key not in self.map_server:
                self.map_server[key] = value                
            else:                
                self.map_server[key] = np.hstack((self.map_server[key], value))
        
        remove_ids = []
        deadtrack_candidates = []
        # Detect dead_tracks
        for feature_id, track in self.map_server.items():
            if track[0, -1] != cur_view_id_:
                remove_ids.append(feature_id)
                deadtrack_candidates.append(track)
            elif track.shape[1] > self.max_slw:
                remove_ids.append(feature_id)
                deadtrack_candidates.append(track)
                
        # Delete deadtrack_candidates
        for feature_id in remove_ids:
            self.map_server.pop(feature_id, None)
                
        # Sort out deadtracks 
        self.dead_tracks = []
        for idx, track in enumerate(deadtrack_candidates):
            if track.shape[1] >= self.min_slw:
                temp_deadtracks = {}
                if track.shape[1] == self.max_slw + 1:
                    temp_deadtracks['feature_id']       = remove_ids[idx]
                    temp_deadtracks['frame']            = track[0, :-1]
                    temp_deadtracks['pts']              = track[1:5, :-1]
                    temp_deadtracks['un_pts']           = track[5:9, :-1]
                else:
                    temp_deadtracks['feature_id']       = remove_ids[idx]
                    temp_deadtracks['frame']            = track[0, :]
                    temp_deadtracks['pts']              = track[1:5, :]
                    temp_deadtracks['un_pts']           = track[5:9, :]
                
                self.dead_tracks.append(temp_deadtracks)                
        return self
                    
                
        