from abc import ABC,abstractmethod
from numpy import ndarray
from typing import Optional

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
    def f(self, x:ndarray, t:Optional[float]=None) -> ndarray:
        '''
        returns drift dynamics
            inputs:
                x: state provided as array of shape (xdims,1)
                t: time for time dependent dynamics. assign any if time invariant
            outputs: 
                f: drift field as array of shape (xdims,1)
        '''
        pass

    @abstractmethod
    def g(self, x:ndarray, t:Optional[float]=None) -> ndarray:
        '''
        returns control affine dynamics
            inputs:
                x: state provided as array of shape (xdims,1)
                t: time for time dependent dynamics. assign any if time invariant
            outputs: 
                g: affine control fields as array of shape (xdims,udims)
        '''
        pass

    @abstractmethod
    def w(self, x:ndarray, t:Optional[float]=None) -> ndarray:
        '''
        returns disturbance on dynamics
        '''
        pass

    @abstractmethod
    def xdot(self,x:ndarray,t:Optional[float]=None,u:Optional[ndarray]=None,noise:bool=False) -> ndarray:
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