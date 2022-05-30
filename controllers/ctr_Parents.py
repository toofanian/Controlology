from abc import ABC,abstractmethod
from numpy import ndarray
from typing import Optional

from systems.sys_Parents import ControlAffineSys

class Controller(ABC):
    '''
    Controller Parent class

    Initialized with a system. Call u(t,x) method to return control for system based on controller algorithm.
    '''
    def __init__(self, sys:ControlAffineSys) -> None:
        super().__init__()
        self.sys:ControlAffineSys = sys()        
        
    @abstractmethod
    def u(self,x:ndarray,t:Optional[float]=None) -> ndarray:
        '''
        returns control u based on current time t and system state x
        inputs: 
            x: ndarray of shape (sys.xDims,1) for current state
            t: float for current time
        outputs: 
            u: ndarray of shape (sys.uDims,1) for current control
        '''
        return


