import numpy as np

from typing import Tuple,Union,Optional
from numpy import ndarray

from .sys_Parents import ControlAffineSys

class activeCruiseControl(ControlAffineSys):
    def __init__(self) -> None:
        xDims = 2
        xBounds = np.array([[0,30],[-100,100]])

        uDims = 1
        uBounds = np.array([[-100000,100000]])

        super().__init__(xDims=xDims,xBounds=xBounds,uDims=uDims,uBounds=uBounds)
            
        self.fric_coeff = [0,0,0]#[0.1, 5., 0.25]
        self.m = 1650
        self.vdes = 10
        self.vlead = 10

    def f(self, x:ndarray, t:Optional[float]=None) -> ndarray:

        def fric(x):
            v = x[0,0]
            fric_coeff = self.fric_coeff
            return fric_coeff[0] + fric_coeff[1]*abs(v) + fric_coeff[2]*(v**2)
        f = np.array([[-1/self.m*fric(x)],[self.vlead - x[0,0]]])
        return f

    def g(self, x:ndarray, t:Optional[float]=None) -> ndarray:

        g = np.array([[1/self.m],[0]])
        return g

    def w(self, x:ndarray, t:Optional[float]=None) -> ndarray:

        w = np.zeros((2,1))
        return w

    def xdot(self,x:ndarray,t:Optional[float]=None,u:Optional[ndarray]=None,noise:bool=False) -> ndarray:
        f = self.f(x,t)
        g = self.g(x,t)
        w = self.w(x,t)
        if noise == False: w = np.zeros(w.shape)
        return f + g@u + w