from abc import ABC,abstractmethod
from types import NoneType
from numpy import ndarray
import torch
from typing import Optional,Union

from systems.sys_Parents import ControlAffineSys

class Controller(ABC):
    '''
    Controller Parent class

    Initialized with a system and optional reference controller. 
    Call u(t,x) method to return control for system based on controller algorithm.
    '''
    def __init__(self, sys:ControlAffineSys, ref:Optional['Controller']=None) -> None:
        super().__init__()
        self.sys:ControlAffineSys = sys
        self.ref:'Controller' = ref
        
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

class nController(Controller):
    '''
    Neural Controller Parent Class
    '''
    def __init__(self,
                 sys:ControlAffineSys,
                 net:torch.nn.Module,
                 ref:Union[Controller,'nController',NoneType]=None) -> None:
        super().__init__(sys=sys,ref=ref)
        self.net = net