from controllers.trainedNetworks.nnTrainer import train_nCLF
from systems.sys_SingleIntegrator import Sys_SingleIntegrator
from visualizers.vis_PlotEpoch import Vis_PlotEpoch
import numpy as np

if __name__ == '__main__':
    '''
    Train and save a neural control lyapunov function (nCLF) for a defined system.

    nCLF is saved as a pytorch ~~~.pth file.
    '''

    # pick a system from /systems/
    sys = Sys_SingleIntegrator()

    # initialize the trainer with the system
    trainer = train_nCLF(sys)
    vis = Vis_PlotEpoch()

    # run the trainer, defining a relative save path
    loss_data = trainer.train(10,100)
    vis.load(loss_data,'10 samples')

    trainer = train_nCLF(sys)
    loss_data = trainer.train(100,100)
    vis.load(loss_data,'100 samples')

    trainer = train_nCLF(sys)
    loss_data = trainer.train(1000,100)
    vis.load(loss_data,'1000 samples')

    vis.render()

