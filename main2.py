from controllers.trainedNetworks.nnTrainer import train_nCLF
from systems.sys_ActiveCruiseControl import activeCruiseControl


if __name__ == '__main__':
    
    trainer = train_nCLF(activeCruiseControl)
    trainer.train()
