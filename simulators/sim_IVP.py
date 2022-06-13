import numpy as np
from scipy.integrate import solve_ivp

from typing import Tuple
from numpy import ndarray

from controllers.ctr_Parents import Controller
from systems.sys_Parents import ControlAffineSys
from simulators.sim_Parents import Simulator

class Sim_SolIVP(Simulator):
    '''
    uses scipy.integrate.solve_ivp to simulate the system
    '''
    def __init__(self,
                 sys: ControlAffineSys,
                 ctr: Controller) -> None:
        super().__init__(sys=sys, ctr=ctr)

    def run(self,
            xInitial:ndarray,
            duration:float,
            noise:bool=False) -> Tuple[ndarray,ndarray]:

        self.u_seq = np.empty((self.sys.uDims+1,0))
        
        def odefunc(t:float,x:ndarray) -> ndarray:
            '''
            helper function for ODE solver.
            '''
            x = np.reshape(x,(self.sys.xDims,1))
            u = self.ctr.u(x=x)

            # store u 
            # BUG: time may not always be sequential in scipy.integrate.solve_ivp
            u_forseq = np.concatenate((u,np.array([[t]])),axis=0)     # stack u with curr time
            self.u_seq = np.concatenate((self.u_seq,u_forseq),axis=1) # append u to u_sequence
            
            xdot = self.sys.xdot(x,t,u,noise=noise)
            return xdot.flatten()

        tspan = (0,duration)
        traj = solve_ivp(odefunc,tspan,xInitial.flatten(),max_step=duration/100)    
        
        x_data = np.block([[traj.y],[np.reshape(traj.t,(1,traj.t.shape[0]))]])
        u_data = self.u_seq
        return x_data,u_data