from abc import ABC,abstractmethod
from numpy import ndarray
from typing import Tuple

from systems.sys_Parents        import ControlAffineSys
from controllers.ctr_Parents    import Controller

class Simulator(ABC):
    def __init__(self,
                 sys:ControlAffineSys,
                 ctr:Controller) -> None:
        super().__init__()
        self.sys:ControlAffineSys = sys()
        self.ctr:Controller = ctr(sys)

    @abstractmethod
    def run(self,IC:ndarray,duration:float,noise:bool=False) -> Tuple[ndarray,ndarray]:
        '''
        run the sim with the initial condition (IC) for the duration (seconds).
        return the results as an (x_dims,t) array and (u_dims,t) array.
        '''
        return
