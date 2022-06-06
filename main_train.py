from controllers.trainedNetworks.nnTrainer import train_nCLF
from systems.sys_SingleIntegrator import Sys_SingleIntegrator


if __name__ == '__main__':
    '''
    to use: 
    import desired system model, and send as arg to train_nCLF class.
    then, run the trainer.
    '''

    # pick a system
    sys = Sys_SingleIntegrator()

    # initialize the trainer
    trainer = train_nCLF(sys)

    # run the trainer, defining a save path if desired
    trainer.train('controllers/trainedNetworks/SingleIntegrator_test2')
