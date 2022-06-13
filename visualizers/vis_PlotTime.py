from types import NoneType
from matplotlib import pyplot as plt
import numpy as np

from .vis_Parents import Visualizer

class Vis_PlotTime(Visualizer):
    def __init__(self) -> None:
        super().__init__()
        self.data_list = []

    def load(self,
             x_data:np.ndarray,
             u_data:np.ndarray,
             label:str):
        self.data_list.append((x_data,u_data,label))

    def render(self):
        colors = ['b','g','r','c','m','k']
        if len(self.data_list) < 10:
            for x_data,u_data,label in self.data_list:
                for i in range(x_data.shape[0]-1):
                    plt.plot(x_data[-1,:],x_data[i,:],label=label+f', x{i+1}')
            plt.xlabel('time')
            plt.legend(loc='best')
        else:
            print('Renderer: too much data. removing legend and colors')
            for x_data,u_data,label in self.data_list:
                ic = 0
                for i in range(x_data.shape[0]-1):
                    plt.plot(x_data[-1,:],x_data[i,:],colors[ic],label=label+f', x{i+1}')
                    ic += 1
            plt.xlabel('time')
        
        plt.show()        