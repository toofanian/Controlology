import numpy as np

from systems.sys_Parents    import ControlAffineSys
from .ctr_Parents           import Controller
from utilities.utils        import reftraj_lsd

class Ctr_P(Controller):
    def __init__(self,sys:ControlAffineSys) -> None:
        super().__init__(sys=sys)

    def u(self, t:float, x:np.ndarray) -> np.ndarray:
        u = np.zeros((self.sys.uDims,1))

        xref = reftraj_lsd(t)
        
        k_v = 0.55
        k_w = 1.0

        angle_diff = xref[2,0] - x[2,0]
        angle_diff = (angle_diff+np.pi)%(2*np.pi) - np.pi

        u[0,0] = k_v*np.sqrt((x[0,0] - xref[0,0])**2 + (x[1,0] - xref[1,0])**2)
        u[1,0] = k_w*angle_diff

        for uDim in range(self.sys.uDims):
            u[uDim,:] = np.clip(u[uDim,:],self.sys.uBounds[uDim,0],self.sys.uBounds[uDim,1])

        return u