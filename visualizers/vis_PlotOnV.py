from types import NoneType
from matplotlib import pyplot as plt
import numpy as np
import torch

from .vis_Parents import Visualizer

class Vis_PlotOnV(Visualizer):
    def __init__(self) -> None:
        super().__init__()
        self.data_list = []

    def loadData(self,
             x_data:np.ndarray,
             label:str):

        x = x_data[:-1,:]
        x_ten = torch.tensor(x.T)
        v_data = self.nclf(x_ten.float()).detach().numpy()

        self.data_list.append((x_data,v_data,label))

    def loadnCLF(self,net:torch.nn.Module):
        self.nclf = net

    def render(self):
        colors = ['b','g','r','c','m','k']
        styles = [':','--']

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        for x_data,v_data,label in self.data_list:
            ax.plot3D(x_data[0,:], x_data[1,:], v_data[:,0], 'gray')
            ax.plot3D(x_data[0,:], x_data[1,:], v_data[:,0], 'gray')
            ax.scatter3D(x_data[0,-1], x_data[1,-1], v_data[-1,0], 'red')

        plt.show()
