import numpy as np
import torch

###############################################################

from systems.sys_SingleIntegrator      import Sys_SingleIntegrator
from controllers.ctr_nCLF              import Ctr_nCLF
from controllers.ctr_CLF_L2            import Controller_CLF_L2
from simulators.sim_IVP                import Sim_SolIVP
from visualizers.vis_PlotTime          import Vis_PlotTime
from visualizers.vis_PlotTime_compare  import Vis_PlotTime_compare

###############################################################

if __name__ == '__main__':

    # choose system
    sys = Sys_SingleIntegrator()

    # choose controller. if neural controller, preload network.
    net = torch.load('controllers/trainedNetworks/SingleIntegrator_test3/epoch200.pth')
    ctrA = Ctr_nCLF(sys=sys,net=net,ref=None)
    ctrB = Controller_CLF_L2(sys=sys,ref=None)

    # choose simulator and visualizer
    simA = Sim_SolIVP(sys=sys,ctr=ctrA)
    simB = Sim_SolIVP(sys=sys,ctr=ctrB)
    vis = Vis_PlotTime_compare()

    # define sim conditions and run
    for xi in np.arange(start=-1,stop=1.1,step=.5):  # try multiple initial conditions
        xInitial = np.array([[xi]])
        duration = 15
        noise = False
        x_data,u_data = simA.run(xInitial=xInitial,duration=duration,noise=False)
        vis.load('A',x_data,u_data,np.array2string(xInitial))
        x_data,u_data = simB.run(xInitial=xInitial,duration=duration,noise=False)
        vis.load('B',x_data,u_data,np.array2string(xInitial))
    
    # choose visualizer and plot data (or use custom script)
    vis.render()
