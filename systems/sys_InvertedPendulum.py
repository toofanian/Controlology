from .sys_Parents import ControlAffineSys
import numpy as np
from typing import Tuple,Union,Optional

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

    def f(self,
          t:float, 
          x:np.ndarray) -> np.ndarray:
        x1 = x[0,0]
        x2 = x[1,0]
        m = self.params['m']
        l = self.params['l']
        b = self.params['b']

        f = np.array([[x1],[  (1/l)*np.sin(x1) - (b/(m*l**2))*x2  ]])
        return f

    def g(self,
          t:float, 
          x:np.ndarray) -> np.ndarray:
        x1 = x[0,0]
        x2 = x[1,0]
        g = self.params['g']
        m = self.params['m']
        l = self.params['l']
        b = self.params['b']
        g = np.array([[0],[1/(m*l**2)]])
        return g

    def w(self,
          t:float, 
          x:np.ndarray) -> np.ndarray:
        w = np.zeros((2,1))
        return w

    def xdot(self, 
             t:float, 
             x:np.ndarray, 
             u:np.ndarray, 
             noise:bool=False) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        f = self.f(t,x)
        g = self.g(t,x)
        w = self.w(t,x)
        if noise == False: w = np.zeros(w.shape)
        return f + g@u + w

