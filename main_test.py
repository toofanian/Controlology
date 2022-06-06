import numpy as np
import torch

###############################################################

from systems.sys_SingleIntegrator           import Sys_SingleIntegrator
from controllers.ctr_nCLF                   import Ctr_nCLF
from simulators.sim_IVP                     import Sim_SolIVP
from visualizers.vis_PlotTime               import Vis_PlotTime

###############################################################

if __name__ == '__main__':
    '''
    to use: 
    import desired system model, controller, simulator, visualizer, and test suite from modules above,
    then run the test suite with args below.
    '''

    # choose system
    sys = Sys_SingleIntegrator()

    # choose controller. if neural controller, preload network.
    net = torch.load('controllers/trainedNetworks/SingleIntegrator_test2/epoch20.pth')
    ctr = Ctr_nCLF(sys=sys,net=net,ref=None)

    # choose simulator
    sim = Sim_SolIVP(sys=sys,ctr=ctr)

    # define sim conditions and run
    xInitial = np.array([[.5]])
    duration = 20
    noise = False
    x_data,u_data = sim.run(xInitial=xInitial,duration=duration,noise=False)
    
    # choose visualizer and plot data (or use custom script)
    vis = Vis_PlotTime()
    vis.render(x_data,u_data)
