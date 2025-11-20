import yaml
import numpy as np

def YAMLReader(dir, config_file_):
    # load data
    config_path = dir + '/config/' + config_file_
    with open(config_path) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)    
        
    lcam_path = dir + '/config/' + configs['cam0_calib']
    with open(lcam_path) as file:
        lcam_config = yaml.load(file, Loader=yaml.FullLoader)    
    
    rcam_path = dir + '/config/' + configs['cam1_calib']
    with open(lcam_path) as file:
        rcam_config = yaml.load(file, Loader=yaml.FullLoader)    
    
    # data processing
    Kl_ = lcam_config['projection_parameters']
    Kr_ = rcam_config['projection_parameters']
    dc_l_ = lcam_config['distortion_parameters']
    dc_r_ = rcam_config['distortion_parameters']
    
    Kl = np.array([[Kl_['fx'], 0, Kl_['cx']],
                   [0, Kl_['fy'], Kl_['cy']],
                   [0, 0, 1]])
    
    Kr = np.array([[Kr_['fx'], 0, Kr_['cx']],
                   [0, Kr_['fy'], Kr_['cy']],
                   [0, 0, 1]])
    
    dc_l = np.array([dc_l_['k1'], dc_l_['k2'], dc_l_['p1'], dc_l_['p2']])
    dc_r = np.array([dc_r_['k1'], dc_r_['k2'], dc_r_['p1'], dc_r_['p2']])
    
    Tbl = np.reshape(configs['body_T_cam0']['data'], (configs['body_T_cam0']['rows'], configs['body_T_cam0']['cols']))
    Tbr = np.reshape(configs['body_T_cam1']['data'], (configs['body_T_cam0']['rows'], configs['body_T_cam0']['cols']))
    Tlr = np.linalg.inv(Tbl) @ Tbr
    
    # data binding
    configs['lcam'] = {'K': Kl, 'Tcb': np.linalg.inv(Tbl), 'Tbc': Tbl, 'Tlr': Tlr, 'Trl': np.linalg.inv(Tlr),
                       'dc' : dc_l, 'dist_model' : lcam_config['model_type'], 'width': lcam_config['image_width'], 'height': lcam_config['image_height']}                    
    configs['rcam'] = {'K' : Kr, 'dc' : dc_r, 'dist_model' : rcam_config['model_type']}
    return configs