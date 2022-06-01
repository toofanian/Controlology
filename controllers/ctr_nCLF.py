import torch
import cvxpy as cp
import numpy as np

from numpy import ndarray
from typing import Optional
from types import NoneType

from .ctr_Parents import Controller
from systems.sys_Parents import ControlAffineSys
from utilities.clfqp import makeproblem_clfqp

class Ctr_nCLF(Controller):
    '''
    IMPORTANT (TODO): sim is not set up to pass a model path, so the default model_path argument ust be overwritten in __init__ manually.
    '''
    def __init__(self,
                 sys: ControlAffineSys,
                 model_path:str = 'controllers/trainedNetworks/singleInt_epoch9.pth',
                 refcontrol:Optional[Controller]=None) -> None: #TODO implement ref control class
        super().__init__(sys)
        self.nCLF = torch.load(model_path)
        self.refcontrol:Controller = refcontrol(sys) if type(refcontrol) != NoneType else np.zeros((self.sys.uDims,1))
        self.clfqp = makeproblem_clfqp(self)

    def u(self,x:ndarray,t:Optional[float]=None) -> ndarray:
        u_ref = self.refcontrol.u(x,t) if type(self.refcontrol) != np.ndarray else self.refcontrol

        f = self.sys.f(x,t)
        g = self.sys.g(x,t)

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