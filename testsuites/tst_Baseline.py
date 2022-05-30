from numpy import ndarray

from systems.sys_Parents        import ControlAffineSys
from controllers.ctr_Parents    import Controller
from simulators.sim_Parents     import Simulator
from visualizers.vis_Parents    import Visualizer
from testsuites.tst_Parents     import TestSuite

class Tst_Baseline(TestSuite):
    def __init__(self,
                 sys:ControlAffineSys,
                 ctr:Controller,
                 sim:Simulator,
                 vis:Visualizer) -> None:
        super().__init__(sys=sys,ctr=ctr,sim=sim,vis=vis)
    
    def run(self,
            IC:ndarray,
            duration:float,
            noise:bool=False,
            verbose:bool=True):
        '''
        sets up simulator with system and controller, runs it with initial condition,
        then sends the simulated data to the visualizer for rendering.
        '''

        assert IC.shape == (self.sys.xDims,1), f'IC shape mismatch. Is {IC.shape}, should be {(self.sys.xDims,1)}'

        self.x_data,self.u_data = self.sim.run(IC,duration,noise)
        if verbose: self.vis.render(self.x_data,self.u_data)
