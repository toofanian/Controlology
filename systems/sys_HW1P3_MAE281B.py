import numpy as np

from typing import Tuple,Union,Optional
from numpy import ndarray

from .sys_Parents import ControlAffineSys

class HW1P3_MAE281B(ControlAffineSys):
    def __init__(self) -> None:
        xDims = 3
        xBounds = np.array([[-3,3],[-3,3]])

        uDims = 1
        uBounds = np.array([[-10000,10000]])

        super().__init__(xDims=xDims,xBounds=xBounds,uDims=uDims,uBounds=uBounds)

    def f(self, x:ndarray, t:Optional[float]=None) -> ndarray:
        f = np.array([[-x[0,0]+x[1,0]],
                      [x[0,0]-x[1,0]-x[0,0]*x[2,0]],
                      [x[0,0]-x[0,0]*x[1,0] - 2*x[2,0]]])
        return f

    def g(self, x:ndarray, t:Optional[float]=None) -> ndarray:
        g = np.array([[0],[1],[0]])
        return g

    def w(self, x:ndarray, t:Optional[float]=None) -> ndarray:
        w = np.zeros((3,1))
        return w

    def xdot(self,x:ndarray,t:Optional[float]=None,u:Optional[ndarray]=None,noise:bool=False) -> ndarray:
        f = self.f(t,x)
        g = self.g(t,x)
        w = self.w(t,x)
        if u == None: u = np.zeros((self.uDims,1))
        if noise == False: w = np.zeros(w.shape)
        return f + g@u + w
