from types import NoneType
from matplotlib import pyplot as plt
import numpy as np

from .vis_Parents import Visualizer

class Vis_PlotTime(Visualizer):
    def __init__(self) -> None:
        super().__init__()
    
    def render(self,x_data:np.ndarray,u_data:np.ndarray):
        def exponential(x0,t):
            return x0*np.e**(-.5*t)

        for i in range(x_data.shape[0]-1):
            plt.plot(x_data[-1,:],x_data[i,:],label=f'x{i+1}')
            #y_exp = exponential(x_data[i,0],x_data[-1,:])
            #plt.plot(x_data[-1,:],y_exp,label='exponential')

        
        plt.xlabel('time [sec]')
        plt.legend(loc='upper right')
        
        
        plt.show()



        
        if type(u_data) != NoneType:
            for i in range(u_data.shape[0]-1):
                plt.plot(u_data[-1,:],u_data[i,:])
            plt.show()