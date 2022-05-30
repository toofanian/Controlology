import numpy as np

from numpy import ndarray
from typing import Optional

from .ctr_Parents           import Controller
from systems.sys_Parents    import ControlAffineSys
from utilities.utils        import reftraj_lsd

class Ctr_CEC(Controller):
    def __init__(self, sys:ControlAffineSys) -> None:
        super().__init__(sys=sys)
    
    def u(self,x:ndarray,t:Optional[float]=None) -> ndarray:
        u = np.zeros((self.sys.uDims,1))
        xref = reftraj_lsd(t)
        return u