from types import NoneType
from matplotlib import pyplot as plt
import numpy as np

from .vis_Parents import Visualizer

class Vis_PlotEpoch(Visualizer):
    def __init__(self) -> None:
        super().__init__()
        self.data_list = []

    def load(self,
             x_data:np.ndarray,
             label:str):
        self.data_list.append((x_data,label))

    def render(self):
        colors = ['b','g','r','c','m','k']
        styles = [':','--']
        ic = 0
        for x_data,label in self.data_list:
            color = colors[ic]
            istyle = 0
            for i in range(x_data.shape[0]-1):
                style = styles[istyle]
                plt.plot(x_data[-1,:],x_data[i,:],color+style,label=label+f', loss{i+1}')
                istyle += 1
            plt.plot(x_data[-1,:],np.sum(x_data[:-1,:],axis=0),color,label=label+f', total loss')
            ic += 1
        plt.xlabel('epoch')
        plt.yscale('log')
        plt.legend(loc='best')
        
        plt.show()        