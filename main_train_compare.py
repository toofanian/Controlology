from controllers.trainedNetworks.nnTrainer import train_nCLF
from systems.sys_FinalP2_MAE281B import Sys_FinalP2_MAE281B
from visualizers.vis_PlotEpoch import Vis_PlotEpoch
import numpy as np

if __name__ == '__main__':
    '''
    Train and save a neural control lyapunov function (nCLF) for a defined system.

    nCLF is saved as a pytorch ~~~.pth file.
    '''

    # pick a system from /systems/
    sys = Sys_FinalP2_MAE281B()

    # initialize the trainer with the system
    trainer = train_nCLF(sys)
    vis = Vis_PlotEpoch()

    # run the trainer, defining a relative save path
    loss_data = trainer.train(10)
    vis.load(loss_data,'10 samples')

    trainer = train_nCLF(sys)
    loss_data = trainer.train(100)
    vis.load(loss_data,'100 samples')

    trainer = train_nCLF(sys)
    loss_data = trainer.train(1000)
    vis.load(loss_data,'1000 samples')

    vis.render()

