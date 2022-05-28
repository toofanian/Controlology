import numpy as np

from .vis_Parents import Visualizer
from utilities.utils import visualize,reftraj_lsd

class Vis_276BPR3(Visualizer):
    def __init__(self) -> None:
        super().__init__()

    def render(self, data) -> None:
        carstates = data[:3,:].T
        
        reftraj = np.zeros(carstates.shape)
        for i,t in np.ndenumerate(data[-1,:]):
            reftraj[i,:] = reftraj_lsd(t)[:,0]

        obstacles = np.array([[-2,-2,0.5], [1,2,0.5]])
        times = np.diff(data[-1,:]).tolist()
        visualize(carstates,reftraj,obstacles,times,times[0],save=False)
