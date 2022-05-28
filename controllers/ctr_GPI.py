from .ctr_Parents import Controller
from systems.sys_Parents import ControlAffineSys
import numpy as np
from utilities.utils import reftraj_lsd

class Ctr_CEC(Controller):
    def __init__(self, dynsys:ControlAffineSys) -> None:
        super().__init__(dynsys)
    
    def u(self, t:float, x:np.ndarray) -> np.ndarray:
        u = np.zeros((self.dynsys.uDims,1))
        xref = reftraj_lsd(t)
        return u