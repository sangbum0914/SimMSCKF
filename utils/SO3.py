import numpy as np
import scipy.linalg 

class SO3:    
    def expmap(phi_):        
        """Rotation matrix to rotation vector : exp mapping
        Args:
            phi : rotation vector
        Returns:
            R_ (np.array): 3x3
        """
        ESP = 1e-12
        np_phi = np.linalg.norm(phi_)  
        N = 10 if np_phi < ESP else float('inf')
    
        phi_x = SO3.skew(phi_)
    
        sp = np.sin(np_phi)
        cp = np.cos(np_phi)
    
        if N == float('inf'):            
            R = np.eye(3) + sp / np_phi * phi_x + (1 - cp) / np_phi**2 * np.dot(phi_x, phi_x)
        else:            
            phi_xn = np.eye(3)
            R = np.eye(3)
            for n in range(1, N + 1):
                phi_xn = np.dot(phi_xn, phi_x) / n
                R += phi_xn
                        
            RtR = np.dot(R.T, R)
            R = np.dot(R, np.linalg.inv(scipy.linalg.sqrtm(RtR)))      
            
        return R
       
    def logmap(R_):    
        """Rotation vector to rotation matrix : log mapping
        Args:
            R_ (np.array): 3x3
        Returns:
            phi : rotation vector
        """
        ESP = 1e-7
        tr = np.trace(R_)         
        dR = R_ - R_.T
        
        if np.abs(3 - tr) < ESP:  # R is near identity matrix (small rotation)
            phi = 0.5 * SO3.unskew(dR)
        elif np.abs(tr + 1) < ESP:  # R is near -I (rotation by pi)
            phi2 = (np.diag(R_) + np.ones(3)) / 2
            phi_ = np.sqrt(phi2)
            phi = phi_ / np.linalg.norm(phi_) * np.pi
        else:
            pi_ = np.arccos((tr - 1) / 2)
            phi_x = pi_ * (R_ - R_.T) / (2 * np.sin(pi_))
            phi = SO3.unskew(phi_x)
        
        return phi

    def skew(vec_):
        """Return a skew-symmetric form of a vector
        Args:
            vec_ (np.array): (3,) vector
        Returns:
            vec_x: skew_symmetric matrix
        """
        vec_x = np.array([[0,           -vec_[2],   vec_[1]],
                          [vec_[2],     0,          -vec_[0]],
                          [-vec_[1],    vec_[0],    0]])
        return vec_x
    
    def unskew(vec_x_):
        if np.allclose(np.diag(vec_x_), 0) and np.allclose(vec_x_ + vec_x_.T, 0):
            vec = np.array([vec_x_[2, 1], vec_x_[0, 2], vec_x_[1, 0]])
            return vec           
        else:   
            raise ValueError('Input matrix is not in SO(3), it is not skew-symmetric !!!')
            