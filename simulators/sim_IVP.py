import numpy as np
from scipy.integrate import solve_ivp

from typing import Tuple
from numpy import ndarray

from controllers.ctr_Parents import Controller
from systems.sys_Parents import ControlAffineSys
from simulators.sim_Parents import Simulator

class Sim_SolIVP(Simulator):
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

    def run(self,IC:ndarray,duration:float,noise:bool=False) -> Tuple[ndarray,ndarray]:
        self.u_seq = np.empty((self.sys.uDims+1,0))
        def odefunc(t:float,x:np.ndarray) -> np.ndarray:
            # helper function for ODE solver.
            assert x.ndim == 1
            x = np.reshape(x,(self.sys.xDims,1))
            u = self.ctr.u(x=x)
            u_forseq = np.concatenate((u,np.array([[t]])),axis=0)
            self.u_seq = np.concatenate((self.u_seq,u_forseq),axis=1)
            xdot = self.sys.xdot(x,t,u,noise=noise)
            return xdot.flatten()

        tspan = (0,duration)
        traj = solve_ivp(odefunc,tspan,IC.flatten(),max_step = duration/100)    
        
        x_data = np.block([[traj.y],[np.reshape(traj.t,(1,traj.t.shape[0]))]])
        u_data = self.u_seq
        return x_data,u_data