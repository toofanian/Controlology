from abc import ABC,abstractmethod
from numpy import ndarray

class Visualizer(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def render(self,data) -> None:
        pass