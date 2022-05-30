from types import NoneType
import torch
import cvxpy as cp
import numpy as np
from typing import Optional, Tuple, Union

from .ctr_Parents import Controller
from systems.sys_Parents import ControlAffineSys

class Ctr_nCLF(Controller):
    def __init__(self,
                 sys: ControlAffineSys,
                 model_path:str = 'controllers/trainedNetworks/singleint_60epoch_10penalty.pth',
                 refcontrol:Optional[Controller]=None) -> None: #TODO implement ref control class
        super().__init__(sys)
        self.nCLF = torch.load(model_path)
        self.refcontrol:Controller = refcontrol(sys) if type(refcontrol) != NoneType else np.zeros((self.sys.uDims,1))
        self.clfqp = self._makeproblem_clfqp()

    def u(self,t,x):
        u_ref = self.refcontrol.u(t,x) if type(self.refcontrol) != np.ndarray else self.refcontrol

        f = self.sys.f(t,x)
        g = self.sys.g(t,x)

        x_ten = torch.tensor(x.T,requires_grad=True)
        clf_value_ten:torch.Tensor = self.nCLF(x_ten.float())
        clf_value_ten.backward()
        clf_value_grad = x_ten.grad

        clf_value = clf_value_ten.detach().numpy()
        Lf_V = (clf_value_grad@f).type_as(clf_value_ten).detach().numpy()
        Lg_V = (clf_value_grad@g).type_as(clf_value_ten).detach().numpy()

        # send V, Lf_V, Lg_V to clfqp, solve for r (must be done one at a time, not as batch)
        self.r_penalty_param.value = np.array([[100]])
        self.u_ref_param.value = u_ref
        self.V_param.value = clf_value
        self.LfV_param.value = Lf_V
        self.LgV_param.value = Lg_V
        self.clfqp.solve()
        
        #assert self.r_var.value < 1e-5, f'r_var value is negative: {self.r_var.value}'
        r_val = self.r_var.value
        u_val = self.u_var.value
        return u_val

    def _makeproblem_clfqp(self) -> cp.Problem:
        '''
        sets up the CLFQP
        '''
        ### define variables and parameters
        self.u_var = cp.Variable((self.sys.uDims,1))
        self.u_ref_param = cp.Parameter((self.sys.uDims,1))
        self.r_var = cp.Variable((1,1),nonneg=True)
        self.r_penalty_param = cp.Parameter((1,1),nonneg=True)
        self.V_param = cp.Parameter((1,1),nonneg=True) 
        self.LfV_param =  cp.Parameter((1,1))
        self.LgV_param = cp.Parameter((1,self.sys.uDims))
        c = 1 # hyperparameter, should be tied to one used during training

        ### define objective
        objective_expression = cp.sum_squares(self.u_var - self.u_ref_param) + cp.multiply(self.r_var,self.r_penalty_param)
        objective = cp.Minimize(objective_expression)

        ### define constraints
        constraints = []
        # control constraints
        for iu in range(self.sys.uDims):
            constraints.append(self.u_var[iu] >= self.sys.uBounds[iu,0])
            constraints.append(self.u_var[iu] <= self.sys.uBounds[iu,1])
        # CLF constraint
        constraints.append(self.LfV_param + self.LgV_param@self.u_var <= -c*self.V_param + self.r_var)

        ### assemble problem
        problem = cp.Problem(objective,constraints)
        assert problem.is_qp(), 'Problem is not qp'
        return problem