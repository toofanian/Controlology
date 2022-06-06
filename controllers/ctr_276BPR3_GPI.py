import numpy as np

from numpy import ndarray
from typing import Optional

from .ctr_Parents           import Controller
from systems.sys_Parents    import ControlAffineSys
from utilities.utils        import reftraj_lsd

class Ctr_276BPR3_GPI(Controller):
    def __init__(self, sys:ControlAffineSys) -> None:
        super().__init__(sys=sys)
        # NOTE: sys is unused here.

        ### define state space
        xDims = 4 # x,y,theta,time
        xBounds = np.array([[-3,3],[-3,3],[-np.pi,np.pi],[0,100]])
        xRes = np.array([10,10,10,200])
        xMap = [np.linspace(start=xBounds[i,0],stop=xBounds[i,1],num=xRes[i]) for i in range(xDims)]

        def coord2x(x_coord):
            x = [xMap[i][x_coord[i]] for i in range(xDims)]
            return np.array(x).reshape((xDims,1))

        ### define control space
        uDims = 2 # velocity, angular velocity
        uBounds = np.array([[0,1],[-1,1]])
        uRes = np.array([10,10])
        uMap = [np.linspace(start=uBounds[i,0],stop=uBounds[i,1],num=uRes[i]) for i in range(uDims)]
        
        def coord2u(u_coord):
            u = [uMap[i][u_coord[i]] for i in range(uDims)]
            return np.array(u).reshape((uDims,1))

        ### define terminal cost
        wP = np.eye(2)
        wO = np.eye(1)
        def terminal_cost(x_coord):
            x = coord2x(x_coord)
            return x[:2].T @ wP @ x[:2]  +  wO @ (1-np.cos(x[2]))**2

        # initialize value function with terminal cost
        value = np.empty(xRes)
        for x_coord in np.ndindex(value.shape):
            value[x_coord] = terminal_cost(x_coord)

        # define stage cost
        xU = np.eye(2)
        def stage_cost(x_coord,u_coord):
            x = coord2x(x_coord)
            u = coord2u(u_coord)
            return u.T @ xU @ u

        # define motion model
        def motion(x_coord,u_coord):
            dt = .5

            x = coord2x(x_coord)
            u = coord2u(u_coord)

            t = x[-1]
            x = x[:2]
            r = reftraj_lsd(t)
            rn = reftraj_lsd(t+dt)
            
            xdot = np.array([[u[0]*np.cos(x[2]+r[2]) - (rn[0] + r[0]) ],  # x
                             [u[0]*np.sin(x[2]+r[2]) - (rn[1] + r[1]) ],  # y
                             [u[1]                   - (rn[2] + r[2]) ],  # theta
                             [1                                       ]]) # time

            rdot = reftraj_lsd(t+dt) - reftraj_lsd(t)

            xn_mean = x + dt*xdot - rdot
            return

        xref = reftraj_lsd(t)


    
    def u(self,x:ndarray,t:Optional[float]=None) -> ndarray:

        return u