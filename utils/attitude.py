import numpy as np 
from .SO3 import SO3

class Attitude:
    def dcm2euler(dcm):
        """ DCM to Euler angle :  R^g_b = R(-phi) * R(-theta) * R(-psi) = eul^g_b
        Args:
            dcm: A 3x3 direction cosine matrix.
        Returns:
            euler: A numpy array containing the Euler angles [roll, pitch, yaw].
        """
        # Calculate pitch using arctan
        pitch = np.arctan(-dcm[2, 0] / np.sqrt(dcm[2, 1]**2 + dcm[2, 2]**2))

        if dcm[2, 0] <= -0.999:
            # Inducing NaN for roll and yaw
            roll = np.nan
            yaw = np.arctan2(dcm[1, 2] - dcm[0, 1], dcm[0, 2] + dcm[1, 1])  # roll when pitch is -90
        elif dcm[2, 0] >= 0.999:
            # Inducing NaN for roll and yaw
            roll = np.nan
            yaw = np.pi + np.arctan2(dcm[1, 2] + dcm[0, 1], dcm[0, 2] - dcm[1, 1])  # roll when pitch is +90
        else:
            roll = np.arctan2(dcm[2, 1], dcm[2, 2])
            yaw = np.arctan2(dcm[1, 0], dcm[0, 0])

        # Create Euler angles array [roll, pitch, yaw]
        euler = np.array([roll, pitch, yaw])
        return euler

    def euler2dcm(eul):    
        """ Euler angle to DCM : eul^g_b = R^g_b = R(-phi) * R(-theta) * R(-psi)
        Args:
            eul (np.array): 3x1
        Returns:
            R: _description_
        """
        if eul.shape[0] != 3:
            raise ValueError(f"Error of the euler angle input type")
        phi_ = eul[0]
        theta_ = eul[1]
        psi_ = eul[2]

        Rz_ = np.array([[np.cos(psi_), -np.sin(psi_), 0.0],
                        [np.sin(psi_),  np.cos(psi_), 0.0],
                        [0.0,           0.0,          1.0]])

        Ry_ = np.array([[np.cos(theta_),  0.0, np.sin(theta_)],
                        [0.0,             1.0, 0.0],
                        [-np.sin(theta_), 0.0, np.cos(theta_)]])

        Rx_ = np.array([[1.0, 0.0,           0.0],
                        [0.0, np.cos(phi_), -np.sin(phi_)],
                        [0.0, np.sin(phi_),  np.cos(phi_)]])
        
        dcm = Rz_ @ Ry_ @ Rx_
        return dcm

    def dcm2quat(dcm):
        """ DCM to quatenion
        Args:
            dcm (np.array) : 3x3
        Returns:
            quat : 4x1
        """
        d = np.diag(dcm)
        q = np.zeros(4)
        
        q[0] = np.sqrt(1/4 * (1 + d[0] + d[1] + d[2]))
        q[1] = np.sqrt(1/4 * (1 + d[0] - d[1] - d[2]))
        q[2] = np.sqrt(1/4 * (1 - d[0] + d[1] - d[2]))
        q[3] = np.sqrt(1/4 * (1 - d[0] - d[1] + d[2]))

        max_idx = np.argmax(q)

        quat = np.zeros(4)
        if max_idx == 0:
            quat[0] = q[0]
            quat[1] = (dcm[2, 1] - dcm[1, 2]) / (4 * quat[0])
            quat[2] = (dcm[0, 2] - dcm[2, 0]) / (4 * quat[0])
            quat[3] = (dcm[1, 0] - dcm[0, 1]) / (4 * quat[0])
        elif max_idx == 1:
            quat[1] = q[1]
            quat[0] = (dcm[2, 1] - dcm[1, 2]) / (4 * quat[1])
            quat[2] = (dcm[1, 0] + dcm[0, 1]) / (4 * quat[1])
            quat[3] = (dcm[0, 2] + dcm[2, 0]) / (4 * quat[1])
        elif max_idx == 2:
            quat[2] = q[2]
            quat[0] = (dcm[0, 2] - dcm[2, 0]) / (4 * quat[2])
            quat[1] = (dcm[1, 0] + dcm[0, 1]) / (4 * quat[2])
            quat[3] = (dcm[2, 1] + dcm[1, 2]) / (4 * quat[2])
        elif max_idx == 3:
            quat[3] = q[3]
            quat[0] = (dcm[1, 0] - dcm[0, 1]) / (4 * quat[3])
            quat[1] = (dcm[0, 2] + dcm[2, 0]) / (4 * quat[3])
            quat[2] = (dcm[2, 1] + dcm[1, 2]) / (4 * quat[3])

        # q0(scalar)가 음수일 경우 부호를 반전시켜서 q0가 양수가 되도록 보정
        if quat[0] < 0:
            quat = -quat

        return quat
        
    def euler2quat(eul):
        """Euler angle to quaternion
        Args:
            eul (np.array): _description_
        Returns:
            quat (np.array) : 4x1    
        """
        R_ = Attitude.euler2dcm(eul)
        quat = Attitude.dcm2quat(R_)
        return quat
        
    def quat2euler(quat):
        """Quaternion to euler angle
        Args:
            quat (np.array) : 4x1
        Returns:
            euler : 3x1
        """
        dcm_ = Attitude.quat2dcm(quat)  # Convert quaternion to DCM
        euler = Attitude.dcm2euler(dcm_)  # Convert DCM to Euler angles
        return euler

    def quat2dcm(quat):
        """ Quaternion to DCM
        Args:
            quat (np.array): 4x1
        Returns:
            DCM: 3x3
        """
        qs = quat[0]        # scalar part        
        q_vec = quat[1:4]   # vector part
        
        dcm = (qs**2 - np.linalg.norm(q_vec)**2) * np.eye(3) + 2 * qs * SO3.skew(q_vec) + 2 * np.outer(q_vec, q_vec)
        return dcm      

    def rvec2quat(rvec):    
        """Rotation vector to quaterion 
        Args:
            rvec (np.array): 3x1
        Raises:
            ValueError: Rotation angle is zero.
        Returns:
            quaternion : 4x1
        """
        quat = np.zeros(4)
        rot_ang = np.linalg.norm(rvec)  # Compute the rotation angle (magnitude of the rotation vector)
        if rot_ang == 0:
            raise ValueError("The rotation vector has zero magnitude. A valid rotation vector is required.")
        else:
            cR = np.cos(rot_ang / 2)  # Cosine of half the rotation angle
            sR = np.sin(rot_ang / 2)  # Sine of half the rotation angle
            k = rvec / rot_ang  # Unit vector in the direction of rotation
            
            quat = np.concatenate(([cR], sR * k))  # Combine the scalar and vector part into a quaternion
        return quat

    def rvec2dcm(rvec):
        """ Rotation vector to a direction cosine matrix (DCM).
        Input:
            rvec (np.array): 3x1
        Output:
            R : 3x3
        """
        theta = np.linalg.norm(rvec)        
        k = rvec / theta
        ks = SO3.skew(k)
        
        # Rodrigues' rotation formula
        R = np.eye(3) + np.sin(theta) * ks + (1 - np.cos(theta)) * np.dot(ks, ks)
        return R

    def quatMultiply(q, p):
        """_summary_ : This function calculates quaternion multiplication (q * p)        
        Args:
            q (np.array) : 4x1 [qs(scalar), q_vec([q1, q2, q3])] 
            p (np.array) : 4x1 [ps(scalar), p_vec([p1, p2, p3])]
        Returns:
            q_result : 4x1 [qs_out(scalar), q_vec_out([p1, p2, p3])]
        """
        
        qs = q[0]               # Scalar part of q
        ps = p[0]               # Scalar part of p
        q_vec = q[1:4]          # Vector part of q
        p_vec = p[1:4]          # Vector part of p

        q_result = np.zeros(4)  # Initialize a 4D result quaternion

        # Scalar part of the result
        q_result[0] = qs * ps - np.dot(q_vec, p_vec)
        if q_result.dtype == 'complex128':
            stop = 1
        
        # Vector part of the result
        q_result[1:4] = qs * p_vec + ps * q_vec + np.cross(q_vec, p_vec)

        # Ensure the scalar part is positive (to avoid ambiguity in quaternion representation)
        if q_result[0] < 0.0:
            q_result = -q_result

        # Normalizing the quaternion
        q_result = q_result / np.linalg.norm(q_result)
        return q_result
 
    def quat_left_comp(q_):
        qs = q_[0]
        qv = q_[1:4]
        
        q_oplus = np.zeros((4, 4))
        q_oplus = np.block([[qs,  -qv],
                            [qv, qs*np.eye(3) + SO3.skew(qv)]])
        return q_oplus
 
    def quat_right_comp(q_):
        qs = q_[0]
        qv = q_[1:4]
        
        q_ominus = np.zeros((4, 4))
        q_ominus = np.block([[qs,  -qv],
                            [qv.reshape(-1, 1), qs*np.eye(3) - SO3.skew(qv)]])
        return q_ominus
 
    def quat_integrate_RK4(q_, wib_b0_, wib_b1_, dt_):
        dq1 = np.hstack([0, wib_b0_])
        dq2 = np.hstack([0, (wib_b0_ + wib_b1_)/2])
        dq3 = np.hstack([0, wib_b1_])
        
        Om1 = Attitude.quat_right_comp(dq1) 
        Om2 = Attitude.quat_right_comp(dq2)
        Om3 = Attitude.quat_right_comp(dq3)
        
        k1= (1/2) * Om1 @ q_                        # q_ * dq1 임.
        k2 = (1/2) * Om2 @ (q_ + (1/2) * k1 * dt_)
        k3 = (1/2) * Om2 @ (q_ + (1/2) * k2 * dt_)
        k4 = (1/2) * Om3 @ (q_ + k3 * dt_)
        
        qgb_hat = q_ + (1/6) * dt_ * (k1 + 2*k2 + 2*k3 + k4)
        qgb_hat = Attitude.quat_normalize(qgb_hat)        
        return qgb_hat
    
    def quat_normalize(q_):
        qn = q_/np.linalg.norm(q_)
        return qn
        
    def align_INS(fb0_, wb0_):
        """Initial two step coarse alignment of inertial navigation system
        Args:
            fb0_ (np.array): Nx3, accel. samples
            wb0_ (np.array): Nx3, gyro. samples

        Returns:
            q0  (np.array): initial attitude
            bg0 (np.array): initial gyro bias             
        """
        fb0_bar = np.mean(fb0_, axis=0)
        wb0_bar = np.mean(wb0_, axis=0)
        
        roll  = np.arctan2(-fb0_bar[1], -fb0_bar[2])
        pitch = np.arctan(fb0_bar[0]/np.sqrt(fb0_bar[1]**2 + fb0_bar[2]**2))
        yaw = 0
        eul = np.array([roll, pitch, yaw])
        
        q0 = Attitude.euler2quat(eul)
        bg0 = np.array(wb0_bar)              
        return q0, bg0