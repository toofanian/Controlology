import numpy as np
import torch

###############################################################

from systems.sys_SingleIntegrator      import Sys_SingleIntegrator
from systems.sys_HW1P3_MAE281B         import Sys_HW1P3_MAE281B
from controllers.ctr_nCLF              import Ctr_nCLF
from controllers.ctr_CLF_L2            import Controller_CLF_L2
from simulators.sim_IVP                import Sim_SolIVP
from visualizers.vis_PlotTime          import Vis_PlotTime
from visualizers.vis_PlotTime_compare  import Vis_PlotTime_compare

###############################################################

if __name__ == '__main__':

    # choose system
    sys = Sys_HW1P3_MAE281B()

    # choose controller. if neural controller, preload network.
    net = torch.load('controllers/trainedNetworks/HW1P3_MAE281B_test3/epoch200.pth')
    ctrA = Ctr_nCLF(sys=sys,net=net,ref=None)
    ctrB = Controller_CLF_L2(sys=sys,ref=None)

    # choose simulator and visualizer
    simA = Sim_SolIVP(sys=sys,ctr=ctrA)
    simB = Sim_SolIVP(sys=sys,ctr=ctrB)
    vis = Vis_PlotTime_compare()

    # define sim conditions and run
    xInitial = np.array([[-1,-.25,1]])
    duration = 30
    noise = False
    x_data,u_data = simA.run(xInitial=xInitial,duration=duration,noise=False)
    vis.load('A',x_data,u_data,np.array2string(xInitial))
    x_data,u_data = simB.run(xInitial=xInitial,duration=duration,noise=False)
    vis.load('B',x_data,u_data,np.array2string(xInitial))
    
    # choose visualizer and plot data (or use custom script)
    vis.render()
