from controllers.trainedNetworks.nnTrainer import train_nCLF
from systems.sys_SingleIntegrator import Sys_SingleIntegrator


if __name__ == '__main__':
    
    trainer = train_nCLF(Sys_SingleIntegrator)
    trainer.train()
