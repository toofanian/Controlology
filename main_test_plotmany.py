import numpy as np
import torch

###############################################################

from systems.sys_FinalP2_MAE281B           import Sys_FinalP2_MAE281B
from controllers.ctr_nCLF                   import Ctr_nCLF
from simulators.sim_IVP                     import Sim_SolIVP
from visualizers.vis_PlotTime               import Vis_PlotTime

###############################################################

if __name__ == '__main__':

    # choose system
    sys = Sys_FinalP2_MAE281B()

    # choose controller. if neural controller, preload network.
    net = torch.load('controllers/trainedNetworks/FinalP2_MAE281B_test3/epoch1000.pth')
    ctr = Ctr_nCLF(sys=sys,net=net,ref=None)

    # choose simulator and visualizer
    sim = Sim_SolIVP(sys=sys,ctr=ctr)
    vis = Vis_PlotTime()

    num_samples = 100
    samples = np.random.uniform(low=sys.xBounds[:,0]*.75,high=sys.xBounds[:,1]*.75,size=(num_samples,sys.xBounds.shape[0]))


    # define sim conditions and run
    tally = 0
    for xInitial in samples:
        duration = 30
        noise = False
        x_data,u_data = sim.run(xInitial=xInitial,duration=duration,noise=False)
        if np.linalg.norm(x_data[:-1,-1]) < 1e-1: tally += 1
        vis.load(x_data,u_data,np.array2string(xInitial))
    accuracy = tally/num_samples
    print(f'accuracy: {accuracy:.2}')

    # choose visualizer and plot data (or use custom script)
    vis.render()
