from controllers.trainedNetworks.nnTrainer import train_nCLF
from systems.sys_SingleIntegrator import Sys_SingleIntegrator
from systems.sys_InvertedPendulum import Sys_InvertedPendulum


if __name__ == '__main__':
    '''
    to use: 
    import desired system model, and send as arg to train_nCLF class.
    then, run the trainer.
    '''

    sys = Sys_SingleIntegrator()
    trainer = train_nCLF(sys)
    trainer.train('controllers/trainedNetworks/SingleIntegrator_test2')
