from abc import ABC,abstractmethod
from numpy import ndarray

class Visualizer(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def render(self,x_data,u_data=None) -> None:
        '''
        input state data and control data. Should be column-wise, indexed with last row as time.
        '''
        pass