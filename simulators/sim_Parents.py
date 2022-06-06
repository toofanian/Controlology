from abc import ABC,abstractmethod
from numpy import ndarray
from typing import Tuple

from systems.sys_Parents        import ControlAffineSys
from controllers.ctr_Parents    import Controller

class Simulator(ABC):
    '''
    Simulator Parent Class
    
    Initialized with a system and a controller. Call run() method to simulate the controlled system.

    Inputs: 
        sys: control affine dynamic system instance
        ctr: controller instance
    '''
    def __init__(self,
                 sys:ControlAffineSys,
                 ctr:Controller) -> None:
        super().__init__()
        self.sys:ControlAffineSys = sys
        self.ctr:Controller = ctr

    @abstractmethod
    def run(self,
            xInitial:ndarray,
            duration:float,
            noise:bool=False) -> Tuple[ndarray,ndarray]:
        '''
        run the sim with the initial condition (IC) for the duration, return the state and control sequence.

        Inputs:
            xInitial: array of size (sys.xDim,1) as starting point for dynsys
            duration: duration of simulation in seconds
            noise:    bool indicating usage of sys.w() noise in simulation
        '''
        return
