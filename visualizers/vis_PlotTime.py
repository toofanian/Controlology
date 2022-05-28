from matplotlib import pyplot as plt
import numpy as np

from .vis_Parents import Visualizer

class Vis_PlotTime(Visualizer):
    def __init__(self) -> None:
        super().__init__()
    
    def render(self,data:np.ndarray):
        for i in range(data.shape[0]-1):
            plt.plot(data[-1,:],data[i,:])
        plt.show()
