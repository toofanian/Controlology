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
            IC,
            duration,
            noise=False,
            verbose=True):
        self.data = self.sim.run(IC,duration,noise)
        if verbose: self.vis.render(self.data)
