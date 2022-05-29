import numpy as np
from typing import Tuple,Union,Optional

from .sys_Parents import ControlAffineSys

class Sys_DiffDrive(ControlAffineSys):
    def __init__(self) -> None:
        xDims = 3 #TODO replace this super init pass with abstract property?
        xBounds = np.array([[-3,3],[-3,3]])
        uDims = 2
        uBounds = np.array([[0,1],[-1,1]]) #TODO return these to 0,1 and -1,1
        super().__init__(xDims=xDims,uDims=uDims,uBounds=uBounds)

    def f(self, t:float, x:np.ndarray) -> np.ndarray:
        f = np.zeros((self.xDims,1))
        return f

    def g(self, t:float, x:np.ndarray) -> np.ndarray:
        g1 = np.array([[np.cos(x[2,0])],[np.sin(x[2,0])],[0]])
        g2 = np.array([[0],[0],[1]])
        g = np.concatenate((g1,g2),axis=1)
        return g

    def w(self, t:float, x:np.ndarray) -> np.ndarray:
        mu, sigma = 0, 0.04 # mean and standard deviation for (x,y)
        w_xy = np.random.normal(mu, sigma, (2,1))
        mu, sigma = 0, 0.004  # mean and standard deviation for theta
        w_theta = np.random.normal(mu, sigma, (1,1))
        w = np.concatenate((w_xy, w_theta),axis=0)
        return w

    def xdot(self, t:float, x:np.ndarray, u:np.ndarray, noise:bool=True) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        f = self.f(t,x)
        g = self.g(t,x)
        w = self.w(t,x)
        if noise == False: w = np.zeros(w.shape)
        return f + g@u + w
