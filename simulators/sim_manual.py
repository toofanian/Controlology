import numpy as np

from controllers.ctr_Parents import Controller
from systems.sys_Parents import ControlAffineSys
from simulators.sim_Parents import Simulator

class Sim_Manual(Simulator):
    '''
    uses scipy.integrate.solve_ivp to simulate system

    Inputs:
        dynsys: control affine dynamic system object
        controller: controller object
        xInitial: array of size (xDim,1) as starting point for dynsys
        uBounds: array of size (uDim,2) as (min,max) bounds for each control
        t_duration: duration of simulation in seconds
        verbose: true if visualizations desired
    
    Outputs:
        if verbose: plot of agent and reference trajectory across simulation
    '''
    def __init__(self,
                 sys: ControlAffineSys,
                 ctr: Controller) -> None:
        super().__init__(sys=sys, ctr=ctr)


    def run(self,
            IC:np.ndarray,
            duration:float,
            noise=False) -> np.ndarray:

        dt = .001
        x_seq = np.empty((self.sys.xDims+1,int(duration/dt)))
        u_seq = np.empty((self.sys.uDims+1,int(duration/dt)))


        t = 0
        x = IC
        for i in range(x_seq.shape[1]-1):
            u = self.ctr.u(0,x)
    
            x_seq[:self.sys.xDims,i] = x[:,0]
            u_seq[:self.sys.uDims,i] = u[:,0]
            x_seq[-1,i] = t
            u_seq[-1,i] = t

            xdot = self.sys.xdot(0,x,u)
            x = x + dt*xdot
            t += dt

        return x_seq,u_seq