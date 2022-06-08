from controllers.trainedNetworks.nnTrainer import train_nCLF
from systems.sys_FinalP2_MAE281B import Sys_FinalP2_MAE281B


if __name__ == '__main__':
    '''
    Train and save a neural control lyapunov function (nCLF) for a defined system.

    nCLF is saved as a pytorch ~~~.pth file.
    '''

    # pick a system from /systems/
    sys = Sys_FinalP2_MAE281B()

    # initialize the trainer with the system
    trainer = train_nCLF(sys)

    # run the trainer, defining a relative save path
    trainer.train('controllers/trainedNetworks/FinalP2_MAE281B_test2')
