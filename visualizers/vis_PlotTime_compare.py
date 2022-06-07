from types import NoneType
from matplotlib import pyplot as plt
import numpy as np

from .vis_Parents import Visualizer

class Vis_PlotTime_compare(Visualizer):
    def __init__(self) -> None:
        super().__init__()
        self.data_list_A = []
        self.data_list_B = []

    def load(self,
             AorB:str,
             x_data:np.ndarray,
             u_data:np.ndarray,
             label:str):
        if AorB == 'A':
            self.data_list_A.append((x_data,u_data,label))
        else:
            self.data_list_B.append((x_data,u_data,label))

    def render(self):
        for i in range(len(self.data_list_A)):
            x_data_A,u_data_A,label_A = self.data_list_A[i]
            x_data_B,u_data_B,label_B = self.data_list_B[i]
            for i in range(x_data_A.shape[0]-1):
                plt.plot(x_data_A[-1,:],x_data_A[i,:],'k-',label='Neural, '+label_A+f', x{i+1}',)
                plt.plot(x_data_B[-1,:],x_data_B[i,:],'k--',label='L2norm^2, '+label_B+f', x{i+1}',)

        plt.xlabel('time')
        plt.legend(loc='best')
        
        plt.show()        