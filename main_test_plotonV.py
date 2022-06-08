import numpy as np
import torch

###############################################################

from systems.sys_FinalP2_MAE281B           import Sys_FinalP2_MAE281B
from controllers.ctr_nCLF                   import Ctr_nCLF
from simulators.sim_IVP                     import Sim_SolIVP
from visualizers.vis_PlotOnV               import Vis_PlotOnV

###############################################################

if __name__ == '__main__':

    # choose system
    sys = Sys_FinalP2_MAE281B()

    # choose controller. if neural controller, preload network.
    net = torch.load('controllers/trainedNetworks/FinalP2_MAE281B_test2/epoch200.pth')
    ctr = Ctr_nCLF(sys=sys,net=net,ref=None)

    # choose simulator and visualizer
    sim = Sim_SolIVP(sys=sys,ctr=ctr)
    vis = Vis_PlotOnV()
    vis.loadnCLF(net)

    # define sim conditions and run
    num_samples = 1
    samples = np.random.uniform(low=sys.xBounds[:,0]*.75,high=sys.xBounds[:,1]*.75,size=(num_samples,sys.xBounds.shape[0]))


    for xInitial in samples:  # try multiple initial conditions
        duration = 50
        noise = False
        x_data,u_data = sim.run(xInitial=xInitial,duration=duration,noise=False)
        vis.loadData(x_data,np.array2string(xInitial))
    
    # choose visualizer and plot data (or use custom script)
    vis.render()
