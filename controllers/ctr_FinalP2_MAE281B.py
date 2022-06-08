import numpy as np

from typing import Tuple,Optional,Union
from numpy import ndarray

from .ctr_Parents           import Controller
from systems.sys_Parents    import ControlAffineSys

class Ctr_FinalP2_MAE281B(Controller):
    def __init__(self, sys: ControlAffineSys) -> None:
        super().__init__(sys)

    def u(self, x: ndarray, t: Optional[float] = None) -> ndarray:
        xinput = x
        x = xinput[0,0]
        y = xinput[1,0]

        LfV = -1/4*(3*x**2 + x**3 + 2*y)*(-9*x**2 + 9*x**3 - 2*y + x*(4 + 6*y))
        LgV = -x + (3*x**2)/2 + y

        u_val = np.array([[0]]) if LgV == 0 else np.array([[-(LfV + (LfV**2 + LgV**4)**.5)/LgV]])

        return u_val

