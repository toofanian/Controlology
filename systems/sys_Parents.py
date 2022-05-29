from abc import ABC,abstractmethod
from numpy import ndarray

class ControlAffineSys(ABC):
    def __init__(self,
                 xDims:int,
                 xBounds:ndarray,
                 uDims:int,
                 uBounds:ndarray) -> None:
        super().__init__()
        self.xDims = xDims
        self.xBounds = xBounds
        self.uDims = uDims
        self.uBounds = uBounds

    @abstractmethod
    def f(self,t:float,x:ndarray) -> ndarray:
        '''
        returns drift dynamics
            inputs:
                t: time for time dependent dynamics. assign any if time invariant
                x: state provided as array of shape (xdims,1)
            outputs: 
                f: drift field as array of shape (xdims,1)
        '''
        pass

    @abstractmethod
    def g(self,t:float,x:ndarray) -> ndarray:
        '''
        returns control affine dynamics
            inputs:
                t: time for time dependent dynamics. assign any if time invariant
                x: state provided as array of shape (xdims,1)
            outputs: 
                g: affine control fields as array of shape (xdims,udims)
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
            inputs:
                t: time for time dependent dynamics. assign any if time invariant
                x: state provided as array of shape (xdims,1)
                u: control provided as array of shape (udims,1)
            outputs: 
                xdot: total system dynamics as array of shape (xdims,1)
        '''
        pass

    def modify_ubounds(self,uBounds:ndarray) -> None:
        self.uBounds = uBounds
