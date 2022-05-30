import numpy as np

from typing import Tuple,Union,Optional
from numpy import ndarray

from .sys_Parents import ControlAffineSys

class Sys_InvertedPendulum(ControlAffineSys):
    '''
    Object to handle inverted pendulum dynamics. Can be used to sample state space
    and to calculate dynamics.
    '''
    def __init__(self):
        xDims = 2
        xBounds = np.array([[-2,2],[-2,2]])
        uDims = 1
        uBounds = np.array([[-300,300]])
        super().__init__(xDims, xBounds, uDims, uBounds)
        self.params = {'m':1, 'l':1, 'b':.01}

    def f(self, x:ndarray, t:Optional[float]=None) -> ndarray:
        x1 = x[0,0]
        x2 = x[1,0]
        m = self.params['m']
        l = self.params['l']
        b = self.params['b']

        f = np.array([[x1],[  (1/l)*np.sin(x1) - (b/(m*l**2))*x2  ]])
        return f

    def g(self, x:ndarray, t:Optional[float]=None) -> ndarray:
        x1 = x[0,0]
        x2 = x[1,0]
        g = self.params['g']
        m = self.params['m']
        l = self.params['l']
        b = self.params['b']
        g = np.array([[0],[1/(m*l**2)]])
        return g

    def w(self, x:ndarray, t:Optional[float]=None) -> ndarray:
        w = np.zeros((2,1))
        return w

    def xdot(self,x:ndarray,t:Optional[float]=None,u:Optional[ndarray]=None,noise:bool=False) -> ndarray:
        f = self.f(t,x)
        g = self.g(t,x)
        w = self.w(t,x)
        if u == None: u = np.zeros((self.uDims,1))
        if noise == False: w = np.zeros(w.shape)
        return f + g@u + w

