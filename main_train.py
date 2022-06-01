from controllers.trainedNetworks.nnTrainer import train_nCLF
from systems.sys_SingleIntegrator import Sys_SingleIntegrator


if __name__ == '__main__':
    '''
    to use: 
    import desired system model, and send as arg to train_nCLF class.
    then, run the trainer.
    '''


    trainer = train_nCLF(Sys_SingleIntegrator)
    trainer.train()
