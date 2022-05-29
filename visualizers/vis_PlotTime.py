from types import NoneType
from matplotlib import pyplot as plt
import numpy as np

from .vis_Parents import Visualizer

class Vis_PlotTime(Visualizer):
    def __init__(self) -> None:
        super().__init__()
    
    def render(self,x_data:np.ndarray,u_data:np.ndarray):
        for i in range(x_data.shape[0]-1):
            plt.plot(x_data[-1,:],x_data[i,:])
        plt.show()
        
        if type(u_data) != NoneType:
            for i in range(u_data.shape[0]-1):
                plt.plot(u_data[-1,:],u_data[i,:])
            plt.show()