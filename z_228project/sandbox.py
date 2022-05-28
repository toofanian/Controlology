import torch
import numpy
import torch.nn as nn

a = torch.tensor([[1.,2.]],requires_grad=True)

class NN_clf(nn.Module):
    '''
    basic MLP nn with 2-64-64-1 architecture. No bias on last layer. Tanh activations. 
    '''
    def __init__(self) -> None:
        super().__init__()
        
        self.CLF = nn.Sequential(
            nn.Linear(2,64),
            nn.Tanh(),

            nn.Linear(64,64),
            nn.Tanh(),
            
            nn.Linear(64,1,bias=False)
        )

    def forward(self,x:torch.Tensor):
        return self.CLF(x)

test = NN_clf()

print(torch.autograd.functional.jacobian(test, a))