from controllers.trainedNetworks.nnTrainer import train_nCLF
from systems.sys_SingleIntegrator import Sys_SingleIntegrator
from systems.sys_InvertedPendulum import Sys_InvertedPendulum
from systems.sys_HW1P3_MAE281B import Sys_HW1P3_MAE281B


if __name__ == '__main__':
    '''
    to use: 
    import desired system model, and send as arg to train_nCLF class.
    then, run the trainer.
    '''

    # pick a system
    sys = Sys_HW1P3_MAE281B()

    # initialize the trainer
    trainer = train_nCLF(sys)

    # run the trainer, defining a save path if desired
    trainer.train('controllers/trainedNetworks/HW1P3_MAE281B_test3')
