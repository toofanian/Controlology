from abc import ABC,abstractmethod
from numpy import ndarray

class ControlAffineSys(ABC):
    def __init__(self,
                 xDims:int,
                 uDims:int,
                 uBounds:ndarray) -> None:
        super().__init__()
        self.xDims = xDims
        self.uDims = uDims
        self.uBounds = uBounds

    @abstractmethod
    def f(self,t:float,x:ndarray) -> ndarray:
        '''
        returns drift dynamics
        '''
        pass

    @abstractmethod
    def g(self,t:float,x:ndarray) -> ndarray:
        '''
        returns control affine dynamics
        '''
        pass

    @abstractmethod
    def w(self,t:float,x:ndarray) -> ndarray:
        '''
        returns disturbance on dynamics
        '''
        pass

    @abstractmethod
    def xdot(self,t:float,x:ndarray,u:ndarray,noise:bool) -> ndarray:
        '''
        returns xdot = f(x) + g(x)u + w(x)
        '''
        pass

    def modify_ubounds(self,uBounds:ndarray) -> None:
        self.uBounds = uBounds
