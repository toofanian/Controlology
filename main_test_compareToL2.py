import numpy as np
import torch

###############################################################

from systems.sys_SingleIntegrator      import Sys_SingleIntegrator
from systems.sys_FinalP2_MAE281B         import Sys_FinalP2_MAE281B
from controllers.ctr_nCLF              import Ctr_nCLF
from controllers.ctr_FinalP2_MAE281B   import Ctr_FinalP2_MAE281B
from simulators.sim_IVP                import Sim_SolIVP
from visualizers.vis_PlotTime          import Vis_PlotTime
from visualizers.vis_PlotTime_compare  import Vis_PlotTime_compare

###############################################################

if __name__ == '__main__':

    # choose system
    sys = Sys_FinalP2_MAE281B()

    # choose controller. if neural controller, preload network.
    net = torch.load('controllers/trainedNetworks/FinalP2_MAE281B_test3/epoch1000.pth')
    ctrA = Ctr_nCLF(sys=sys,net=net,ref=None)
    ctrB = Ctr_FinalP2_MAE281B(sys=sys)

    # choose simulator and visualizer
    simA = Sim_SolIVP(sys=sys,ctr=ctrA)
    simB = Sim_SolIVP(sys=sys,ctr=ctrB)
    vis = Vis_PlotTime_compare()

    num_samples = 2
    samples = np.random.uniform(low=sys.xBounds[:,0]*.75,high=sys.xBounds[:,1]*.75,size=(num_samples,sys.xBounds.shape[0]))

    # define sim conditions and run
    for xInitial in samples:
        duration = 10
        noise = False
        x_data,u_data = simA.run(xInitial=xInitial,duration=duration,noise=False)
        vis.load('A',x_data,u_data,np.array2string(xInitial))
        x_data,u_data = simB.run(xInitial=xInitial,duration=duration,noise=False)
        vis.load('B',x_data,u_data,np.array2string(xInitial))
    
    # choose visualizer and plot data (or use custom script)
    vis.render()
