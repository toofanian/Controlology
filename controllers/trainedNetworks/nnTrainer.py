import time
import torch
import torch.nn as nn
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from torch.utils.data import Dataset,DataLoader

from typing import Optional

from systems.sys_Parents import ControlAffineSys

class nCLF(nn.Module):
    '''
    MLP network of shape xdim-64-64-1, with tanh activations.
    Final output is squared for positive definiteness.

    This will represent the neural CLF, once trained.
    '''
    def  __init__(self,
                  x_dims,
                  hidden_size:int=64,
                  hidden_layers:int=2) -> None:
        super().__init__()

        assert hidden_layers > 0

        layers = []
        layers.append(nn.Linear(x_dims,hidden_size)) 
        layers.append(nn.Tanh())
        for _ in range(hidden_layers-1):
            layers.append(nn.Linear(hidden_size,hidden_size))
            layers.append(nn.Tanh())

        self.stack_mlp = nn.Sequential(*layers)

    def forward(self,x):
        out:torch.Tensor = self.stack_mlp(x)     
        out = .5*(out*out).sum(dim=1)[:,None]
        
        # TODO squared to enforce pos def, but is there a better way to do this?
        # https://arxiv.org/pdf/2001.06116.pdf section 3.1 (recommended by Prof Yuanyuan Shi)
        return out


class dataset_UniformStates(Dataset):
    '''
    dataset of num_samples state sample, pulled uniformly from within xBounds. Samples are by column.
    
    xBounds is a (xdim,2) array.
    '''
    def __init__(self,xBounds:np.ndarray,num_samples:int):
        self.data = np.random.uniform(low=xBounds[:,0],high=xBounds[:,1],size=(num_samples,xBounds.shape[0])).T

    def __len__(self) -> int:
        return self.data.shape[1]

    def __getitem__(self, idx:int) -> np.ndarray:
        return self.data[:,idx]


class train_nCLF():
    '''
    Trains a neural clf for a given dynamic system.

    Call train() method to train.
    '''
    def __init__(self,
                 sys:ControlAffineSys) -> None:

        self.sys:ControlAffineSys = sys
        self.network = nCLF(self.sys.xDims)

        ##### setup CLFQP using differentiable cvxpylayers

        ### define variables and parameters
        self.u_var           = cp.Variable((self.sys.uDims,1))  # control vector
        self.r_var           = cp.Variable((1,1),nonneg=True)   # CLF relaxation
        self.r_penalty_param = cp.Parameter((1,1),nonneg=True)  # objective weight for CLF relaxation
        self.V_param         = cp.Parameter((1,1),nonneg=True)  # CLF value
        self.LfV_param       = cp.Parameter((1,1))              # Lie derivative of CLF along drift
        self.LgV_param       = cp.Parameter((1,self.sys.uDims)) # Lie derivative of CLF along control (for each control)
        self.c = 1                                              # scale for exponential stability, not param so prob is dpp

        ### define objective
        objective_expression = 0*cp.sum_squares(self.u_var) + cp.multiply(self.r_var,self.r_penalty_param)
        # BUG r would be > 0 before u was maxed out. Difficult to find correct weight that didnt break solver.
        # As workaround, set control weight to zero. Now r is only > 0 when u is maxed.
        objective = cp.Minimize(objective_expression)

        ### define constraints
        constraints = []
        # control constraints
        for iu in range(self.sys.uDims):
            constraints.append(self.u_var[iu] >= self.sys.uBounds[iu,0]) # lower bound for control u_i
            constraints.append(self.u_var[iu] <= self.sys.uBounds[iu,1]) # upper bound for control u_i
        # CLF constraint
        constraints.append(self.LfV_param + self.LgV_param@self.u_var <= -self.c*self.V_param + self.r_var) # exponential clf condition

        ### assemble problem
        problem = cp.Problem(objective,constraints)
        assert problem.is_dpp(), 'Problem is not dpp'

        # convert to differentiable layer, store in self for later use
        variables = [self.u_var,self.r_var]
        parameters = [self.r_penalty_param,self.V_param,self.LfV_param,self.LgV_param]
        self.clfqp_layer = CvxpyLayer(problem=problem,variables=variables,parameters=parameters) # differentiable solver class


    def train(self,n_samples:int,save:Optional[str]=None):
        num_samples = n_samples

        optimizer = torch.optim.Adam(self.network.parameters(),lr=.002)

        lagrange_atzero = 10.
        lagrange_relax = 100.

        epochs = 100 # Saves at every 10th epoch.
        print(f'training start...')
        loss_at_epoch = np.empty((3,epochs))
        for epoch in range(1,epochs+1):
            # NOTE: resampling data every epoch really helps apparently
            dataset = dataset_UniformStates(xBounds=self.sys.xBounds,num_samples=num_samples)
            dataloader = DataLoader(dataset=dataset,batch_size=1,shuffle=True)
            # TODO allow batch > 1. currently bottlenecked at qp.

            # initialize some training metrics
            ti = time.time()
            average_r = 0

            loss_term1 = torch.zeros((1)) # zero at zero
            loss_term2 = torch.zeros((1)) # vdot neg def

            clf_val_atzero:torch.Tensor = self.network(torch.zeros((self.sys.xDims,1)).T.float())
            loss_term1 += torch.squeeze(lagrange_atzero * clf_val_atzero)

            for i,x_ten in enumerate(dataloader):
                
                # get system dynamics from sys class
                x = x_ten.detach().numpy().T
                f = self.sys.f(x)
                g = self.sys.g(x)
                
                # get clf value and gradient on x
                x_ten.requires_grad_(requires_grad=True)
                clf_value_ten:torch.Tensor = self.network(x_ten.float())
                clf_value_ten.backward(retain_graph=True)
                clf_value_grad = x_ten.grad

                # compute lie derivatives
                LfV_ten:torch.Tensor = (clf_value_grad @ f).type_as(clf_value_ten)
                LgV_ten:torch.Tensor = (clf_value_grad @ g).type_as(clf_value_ten)
                
                # assign penalty for relaxation, should be related to infinity norm of u_bounds
                # BUG currently set up to ignore control effort, will use maximal control,
                # relax penalty is arbitrary since there is nothing to compete with.
                relax_penalty = torch.tensor([[1]],dtype=torch.float32)

                # compute clfqp layer
                params = [relax_penalty,clf_value_ten,LfV_ten,LgV_ten]
                u,r = self.clfqp_layer(*params,solver_args={"max_iters": 500000})

                # BUG r should be nonneg, but sometimes it's just not... so i'm adding a relu...
                # this appears to happen when u=0 is sufficient for dV/dt < 0, so r should be zero anyway.
                # assert r >= -1e-5, f'r is negative: {r}'
                r = torch.relu(r)

                average_r += r.detach().numpy()[0]/num_samples # for reference
                loss_term2 += torch.squeeze(lagrange_relax/num_samples * r)
            
            loss = loss_term1+loss_term2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_at_epoch[0,epoch-1] = loss_term1.detach().numpy()
            loss_at_epoch[1,epoch-1] = loss_term2.detach().numpy()
            loss_at_epoch[-1,epoch-1] = epoch

            print(f'epoch {epoch}\tof {epochs} complete.\tloss: {loss.detach().numpy()[0]}.\taverage r: {average_r}\tepoch time: {round(time.time()-ti)} seconds')
            
            if epoch%10 == 0:
                if type(save) == str:
                    torch.save(self.network,save+f'/epoch{epoch}.pth') 
            
        return loss_at_epoch
