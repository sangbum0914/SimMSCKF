from .attitude import Attitude
from .SO3 import SO3

class NavState:
    def __init__(self, q_, p_, v_, ba_, bg_, gg_):
        """ init NavState
        Args:
            q_ (np.array): 4x1
            p_ (np.array): 3x1
            v_ (np.array): 3x1
            ba_ (np.array): 3x1
            bg_ (np.array): 3x1
            gg (np.array): 3x1
        """
        self.q = q_
        self.p = p_
        self.v = v_
        self.ba = ba_
        self.bg = bg_
        self.gg = gg_
        
    def update_from_global_del_Nav(self, del_Nav_):
        self.q = Attitude.quatMultiply(del_Nav_.q, self.q)
        self.R = Attitude.quat2dcm(self.q)
        self.phi = SO3.logmap(self.R)
        
        self.p = self.p + del_Nav_.p
        self.v = self.v + del_Nav_.v
        self.ba = self.ba + del_Nav_.ba
        self.bg = self.bg + del_Nav_.bg
        self.gg = self.gg + del_Nav_.gg
        Xi_hat = self
        return Xi_hat
        