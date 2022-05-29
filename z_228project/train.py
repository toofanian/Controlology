# ML
from statistics import linear_regression
import torch
import torch.nn as nn
import torch.optim as no
import torch.functional as nf
from torch.utils.data import Dataset,DataLoader
import time

# Convex Optimization
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

# Math/Data
import numpy as np
np.random.seed(124)

# Standard Python
import sys
from typing import Optional, Tuple, Union

####################################################################################

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
            
            nn.Linear(64,1,bias=False),
        )

    def forward(self,x:torch.Tensor):
        x = self.CLF(x)
        x = torch.square(x) # to enforce V > 0 
        # does squaring allow 'bad' discontinuities, since self.CLF is not a unique mapping to the output x?
        return x

class InvertedPendulum():
    '''
    Object to handle inverted pendulum dynamics. Can be used to sample state space
    and to calculate dynamics.
    '''
    def __init__(self,params:Optional[dict]=None) -> None:
        
        # define system dynamics parameters
        self.params = {'g':9.81, 'm':1, 'l':1, 'b':.1}
        if type(params) == dict:
            for key,value in params.items():
                assert key in self.params.keys(), 'invalid param provided: bad key'
                assert value > 0,                 'invalid param provided: value < 0 '

                self.params[key] = params[key]

    def sampleSpace(self,num_samples:int) -> np.ndarray:
        '''
        generates state space samples, formulated for inverted pendulum system.

        input: 
            num_samples: n number of samples desired
        output:
            (2,n) ndarray of samples pulled uniformly from inverted pendulum space
        '''
        return np.random.uniform(low=[-2,-2],high=[2,2],size=(num_samples,2)).T

    def getDynamics(self,x:Union[torch.Tensor,np.ndarray]) -> torch.Tensor:
        '''
        calculates the inverted pendulum system dynamics

        inputs:
            x: (n,2) array of states where n is number of samples
        outputs: 
            f: (n,2) array of control independent dynamics
            g: (n,2) array of control dependent dynamics (affine system)

        '''
        if type(x) == np.ndarray:
            if x.ndim == 1: x = x.reshape((1,2))
            num_samples = x.shape[0]
            g,m,l,b = self.params['g'],self.params['m'],self.params['l'],self.params['b']

            f = np.zeros((num_samples,2))

            f[:,0] = x[:,1]
            f[:,1] = ( (g/l) * np.sin(x[:,0]) ) - ( (b/(m*l**2)) * x[:,1] )

            g = np.zeros((num_samples,2))
            g[:,0] = 0
            g[:,1] = ( 1/(m*l**2) )

            return f,g

        elif type(x) == torch.Tensor:
            if x.ndim == 1: x = x.reshape((1,2))
            num_samples = x.shape[0]
            g,m,l,b = self.params['g'],self.params['m'],self.params['l'],self.params['b']

            f = torch.zeros((num_samples,2)).type_as(x)
            f[:,0] = x[:,1]
            f[:,1] = ( (g/l) * torch.sin(x[:,0]) ) - ( (b/(m*l**2)) * x[:,1] )

            g = torch.zeros((num_samples,2)).type_as(x)
            g[:,0] = 0
            g[:,1] = ( 1/(m*l**2) )

            return f,g
        
        else: Exception(f'Type {type(x)} not supported')

class dataset_dynsys(Dataset):
    '''
    general dataset for n dimensional dynamic system
    
    currently hardcoded for the inverted pendulum
    '''
    def __init__(self,dynsys:InvertedPendulum,num_samples:int):
        self.dynsys:InvertedPendulum = dynsys()
        self.data = self.dynsys.sampleSpace(num_samples)

    def __len__(self) -> int:
        return self.data.shape[1]

    def __getitem__(self, idx:int) -> np.ndarray:
        return self.data[:,idx]

class train_nCLF():
    '''
    neural clf, hardcoded for inverted pendulum system
    '''
    def __init__(self,dynsys:InvertedPendulum) -> None:
        self.dynsys:InvertedPendulum = dynsys
        self.dynamics:InvertedPendulum = self.dynsys()
        self.layer_clfqp = makeProblem_CLFQP()
        self.layer_clf = NN_clf()
        pass

    def train(self):
        num_samples = 5000
        dataset = dataset_dynsys(self.dynsys,num_samples)
        dataloader = DataLoader(dataset=dataset,batch_size=1,shuffle=True) #TODO allow batch>1

        optimizer = no.SGD(self.layer_clf.parameters(), lr=.003)

        epochs = 20
        tic = time.time()
        for epoch in range(epochs):
            if epoch > 0: print(f'trained: epoch {epoch} of {epochs},  \
    running avg epoch time: {(time.time()-tic)/(60*epoch)} minutes,  \
    loss is {loss} ')


            clf_val_0:torch.Tensor = self.layer_clf(torch.tensor([[0,0]]).float()) # ...at derised stable point
            assert clf_val_0 >= 0 

            loss = torch.zeros((1,1))
            loss += 10*clf_val_0
            for i,x in enumerate(dataloader):
                # calculate dynamics from input batch
                f,g = self.dynamics.getDynamics(x) 

                x.requires_grad_(requires_grad=True)
                # calculate CLF value...
                clf_val:torch.Tensor = self.layer_clf(x.float())                       # ...from input batch

                # with dynamics and CLF value, compute lie derivatives
                # TODO: compute for batch > 1
                clf_val.backward(retain_graph=True)
                Lf_V = (x.grad@f.T).type_as(clf_val)
                Lg_V = (x.grad@g.T).type_as(clf_val)
                x.requires_grad_(requires_grad=False)

                # send V, Lf_V, Lg_V to clfqp, solve for r (must be done one at a time, not as batch)
                # TODO: vectorize this, if supported by cvxopt?
                u_ref = torch.tensor([0.],dtype=torch.float32)
                relax_penalty = torch.tensor([100.],dtype=torch.float32)

                params = [u_ref,relax_penalty,clf_val,Lf_V,Lg_V]
                u,r = self.layer_clfqp(*params)
                #r = torch.relu(r) #BUG: came out negative sometimes, so i hit it with a relu...
                assert r >= -1e-06, 'r = {}'.format(r)  #BUG: r comes out negative

                loss += 100/num_samples * r
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss <= .05: break
        torch.save(self.layer_clf,'trainedCLF.pth')

def makeProblem_CLFQP():
    '''
    sets up the CLFQP, and returns it as a differentiable cvxpylayer

    currently hardcoded for inverted pendulum dynamics
    '''
    ### define objective
    u_var = cp.Variable(1)
    r_var = cp.Variable(1,nonneg=True)
    u_ref_param = cp.Parameter(1)
    r_penalty_param = cp.Parameter(1,nonneg=True)

    objective_expression = cp.sum_squares(u_var - u_ref_param) + cp.multiply(r_var,r_penalty_param)
    objective = cp.Minimize(objective_expression)

    ### define constraints
    constraints = []
    # control constraints
    u_bounds = np.array([[-300,300]])
    constraints.append(u_var >= u_bounds[0,0])
    print(constraints)
    constraints.append(u_var <= u_bounds[0,1])
    print(constraints)

    # # relaxation constraints
    constraints.append(r_var >= 0)

    # CLF constraints
    V_param = cp.Parameter(1,nonneg=True) 
    LfV_param =  cp.Parameter(1)
    LgV_param = cp.Parameter(1)
    c = 1
    constraints.append(LfV_param + LgV_param*u_var + c*V_param - r_var <= 0)

    ### assemble problem
    problem = cp.Problem(objective,constraints)
    assert problem.is_dpp(), 'Problem is not DPP'
    variables = [u_var,r_var]
    parameters = [u_ref_param,r_penalty_param,V_param,LfV_param,LgV_param]

    # convert to differentiable layer, store in self for later use
    return CvxpyLayer(problem=problem,parameters=parameters,variables=variables)
    
    # BUG: training QP doesnt seem to be working right. 
    # u is very small, and r is negative sometimes.


if __name__ == '__main__':
    clf = train_nCLF(InvertedPendulum)
    x = clf.train()
