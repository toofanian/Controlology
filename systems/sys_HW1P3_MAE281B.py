from .sys_Parents import ControlAffineSys
import numpy as np
from typing import Tuple,Optional,Union

class HW1P3_MAE281B(ControlAffineSys):
    def __init__(self) -> None:
        xDims = 3
        xBounds = np.array([[-3,3],[-3,3]])

        uDims = 1
        uBounds = np.array([[-10000,10000]])

        super().__init__(xDims=xDims,xBounds=xBounds,uDims=uDims,uBounds=uBounds)

    def f(self, t:float, x:np.ndarray) -> np.ndarray:
        f = np.array([[-x[0,0]+x[1,0]],
                      [x[0,0]-x[1,0]-x[0,0]*x[2,0]],
                      [x[0,0]-x[0,0]*x[1,0] - 2*x[2,0]]])
        return f

    def g(self, t:float, x:np.ndarray) -> np.ndarray:
        g = np.array([[0],[1],[0]])
        return g

    def w(self, t:float, x:np.ndarray) -> np.ndarray:
        w = np.zeros((3,1))
        return w

    def xdot(self, t:float, x:np.ndarray, u:np.ndarray, noise:bool=True) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        f = self.f(t,x)
        g = self.g(t,x)
        w = self.w(t,x)
        if noise == False: w = np.zeros(w.shape)
        return f + g@u + w

    def modify_ubounds(self,uBounds:np.ndarray):
        self.uBounds = uBounds