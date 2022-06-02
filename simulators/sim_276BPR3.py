import numpy as np
from time import time

from numpy import ndarray
from typing import Tuple

from .sim_Parents               import Simulator
from systems.sys_Parents        import ControlAffineSys
from controllers.ctr_Parents    import Controller

from utilities.utils import reftraj_lsd

class Sim_276BPR3(Simulator):
    def __init__(self, 
                 sys: ControlAffineSys, 
                 ctr: Controller) -> None:
        super().__init__(sys, ctr)

    def run(self,IC:ndarray,duration:float,noise:bool=False) -> Tuple[ndarray,ndarray]:
        time_step = 0.5        # time between steps in seconds
        sim_time = duration    # simulation time
        t = 0

        traj = np.zeros((self.sys.xDims+1,int(sim_time/time_step)+1))
        reftraj = np.zeros((self.sys.xDims+1,int(sim_time/time_step)+1))
        u_seq = np.zeros((self.sys.uDims,int(sim_time/time_step)))

        x = IC
        i = 0
        traj[:self.sys.xDims,i] = x[:,0]
        reftraj[:self.sys.xDims,i] = reftraj_lsd(t)[:,0]

        error = 0.
        timer_main_loop_start = time()
        times_inner_loop = np.zeros((int(sim_time/time_step)))
        while t < sim_time:
            t1 = time()
            i += 1

            u = self.ctr.u(x,t)
            u_seq[:,i-1] = u[:,0]
            x += time_step * self.sys.xdot(x,t,u,noise=noise)
            t += time_step
            traj[:self.sys.xDims,i] = x[:,0]
            traj[-1,i] = t
            reftraj[:self.sys.xDims,i] = reftraj_lsd(t)[:,0]
            error += np.linalg.norm(traj[:self.sys.xDims,i]-reftraj[:self.sys.xDims,i])

            times_inner_loop[i-1] = time()-t1
        timer_main_loop_end = time()

        print(f'Total time: {timer_main_loop_end-timer_main_loop_start}')
        print(f'Average iteration time: {times_inner_loop.mean()*1000} ms')
        print('Final error: ', error)

        return traj
        

