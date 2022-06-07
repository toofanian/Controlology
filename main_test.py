import numpy as np
import torch

###############################################################

from systems.sys_SingleIntegrator           import Sys_SingleIntegrator
from controllers.ctr_nCLF                   import Ctr_nCLF
from simulators.sim_IVP                     import Sim_SolIVP
from visualizers.vis_PlotTime               import Vis_PlotTime

###############################################################

if __name__ == '__main__':

    # choose system
    sys = Sys_SingleIntegrator()

    # choose controller. if neural controller, preload network.
    net = torch.load('controllers/trainedNetworks/SingleIntegrator_test3/epoch200.pth')
    ctr = Ctr_nCLF(sys=sys,net=net,ref=None)

    # choose simulator and visualizer
    sim = Sim_SolIVP(sys=sys,ctr=ctr)
    vis = Vis_PlotTime()

    # define sim conditions and run
    xInitial = np.array([[.5]])
    duration = 17.5
    noise = False
    x_data,u_data = sim.run(xInitial=xInitial,duration=duration,noise=False)
    vis.load(x_data,u_data,np.array2string(xInitial))
    
    # choose visualizer and plot data (or use custom script)
    vis.render()
