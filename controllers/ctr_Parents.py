from abc import ABC,abstractmethod
from numpy import ndarray

from systems.sys_Parents import ControlAffineSys

class Controller(ABC):
    def __init__(self, sys:ControlAffineSys) -> None:
        super().__init__()
        self.sys:ControlAffineSys = sys()        
        
    @abstractmethod
    def u(self,t:float,x:ndarray) -> ndarray:
        pass


