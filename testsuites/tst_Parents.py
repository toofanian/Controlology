from abc import ABC,abstractmethod

from systems.sys_Parents import ControlAffineSys
from controllers.ctr_Parents import Controller
from simulators.sim_Parents import Simulator
from visualizers.vis_Parents import Visualizer

class TestSuite(ABC):
    def __init__(self,
                 sys:ControlAffineSys,
                 ctr:Controller,
                 sim:Simulator,
                 vis:Visualizer) -> None:
        super().__init__()
        self.sys:ControlAffineSys = sys()
        self.ctr:Controller = ctr(sys=sys)
        self.sim:Simulator = sim(sys=sys,ctr=ctr)
        self.vis:Visualizer = vis()

    @abstractmethod
    def run(self,
            IC,
            duration,
            noise:bool = False,
            verbose:bool = True):
        '''
        accepts initial condition, duration of simulation, and generates visualizations if verbose
        '''
        return
