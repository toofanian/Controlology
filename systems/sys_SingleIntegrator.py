import numpy as np

from typing import Tuple,Union,Optional
from numpy import ndarray

from .sys_Parents import ControlAffineSys

class Sys_SingleIntegrator(ControlAffineSys):
    def __init__(self) -> None:
        xDims = 1
        xBounds = np.array([[-3,3]])
        uDims = 1
        uBounds = np.array([[-2,2]])

        super().__init__(xDims=xDims,xBounds=xBounds,uDims=uDims,uBounds=uBounds)

    def f(self, x:ndarray, t:Optional[float]=None) -> ndarray:
        f = np.array([[0]])
        return f

    def g(self, x:ndarray, t:Optional[float]=None) -> ndarray:
        g = np.array([[1]])
        return g

    def w(self, x:ndarray, t:Optional[float]=None) -> ndarray:
        w = np.zeros((self.xDims,1))
        return w

    def xdot(self,x:ndarray,t:Optional[float]=None,u:Optional[ndarray]=None,noise:bool=False) -> ndarray:
        f = self.f(x,t)
        g = self.g(x,t)
        w = self.w(x,t)
        if u == None: u = np.zeros((self.uDims,1))
        if noise == False: w = np.zeros(w.shape)
        return f + g@u + w