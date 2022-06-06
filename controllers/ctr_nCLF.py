import torch
import cvxpy as cp
import numpy as np

from numpy import ndarray
from typing import Optional,Union
from types import NoneType

from .ctr_Parents import Controller,nController
from systems.sys_Parents import ControlAffineSys

class Ctr_nCLF(nController):
    '''
    Uses a trained neural network to estimate lyapunov function, then solves then
    solves a CLFQP to return a goal-reaching control.

    r_penalty_param may need adjusting, depending on control and state magnitudes.
    '''

    def __init__(self, 
                 sys: ControlAffineSys, 
                 net: torch.nn.Module, 
                 ref: Union[Controller, nController, NoneType] = None) -> None:
        super().__init__(sys=sys, net=net, ref=ref)

        ###### set up the CLFQP problem using cvxpy

        ### define variables and parameters
        self.u_var           = cp.Variable((self.sys.uDims,1))  # control vector
        self.u_ref_param     = cp.Parameter((self.sys.uDims,1)) # refrence control vector, if used
        self.r_var           = cp.Variable((1,1),nonneg=True)   # CLF relaxation
        self.r_penalty_param = cp.Parameter((1,1),nonneg=True)  # objective weight for CLF relaxation
        self.V_param         = cp.Parameter((1,1),nonneg=True)  # CLF value
        self.LfV_param       = cp.Parameter((1,1))              # Lie derivative of CLF along drift
        self.LgV_param       = cp.Parameter((1,self.sys.uDims)) # Lie derivative of CLF along control (for each control)
        self.c = 1                                              # scale for exponential stability, not param so prob is dpp
        
        ### define objective
        objective_expression = ( cp.sum_squares(self.u_var - self.u_ref_param)        # use control nearest to 
                                    + cp.multiply(self.r_var,self.r_penalty_param) )
        objective = cp.Minimize(objective_expression)

        ### define constraints
        constraints = []
        # control constraints
        for iu in range(self.sys.uDims):
            constraints.append(self.u_var[iu] >= self.sys.uBounds[iu,0])
            constraints.append(self.u_var[iu] <= self.sys.uBounds[iu,1])
        # CLF constraint
        constraints.append(self.LfV_param + self.LgV_param@self.u_var <= -self.c*self.V_param + self.r_var)

        ### assemble problem
        self.problem = cp.Problem(objective,constraints)

    def u(self,x:ndarray,t:Optional[float]=None) -> ndarray:

        ### get clf value,grad and compute lie derivatives
 
        x_ten = torch.tensor(x.T,requires_grad=True)
        clf_value_ten:torch.Tensor = self.net(x_ten.float())
        clf_value_ten.backward()
        clf_value_grad = x_ten.grad
        clf_value = clf_value_ten.detach().numpy()

        f = self.sys.f(x,t)
        g = self.sys.g(x,t)
        Lf_V = (clf_value_grad@f).type_as(clf_value_ten).detach().numpy()
        Lg_V = (clf_value_grad@g).type_as(clf_value_ten).detach().numpy()

        # assign clfqp parameter values
        self.u_ref_param.value      = self.ref.u(x,t) if type(self.ref) == Controller else np.zeros((self.sys.uDims,1))
        self.r_penalty_param.value  = np.array([[100]]) #TODO make dynamic?
        self.V_param.value          = clf_value
        self.LfV_param.value        = Lf_V
        self.LgV_param.value        = Lg_V

        self.problem.solve(solver=cp.ECOS)
        
        r_val = self.r_var.value
        u_val = self.u_var.value
        return u_val

