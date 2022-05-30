from .ctr_Parents import Controller
from systems.sys_Parents import ControlAffineSys
import numpy as np
import cvxpy as cp
from typing import Tuple,Optional,Union

from utilities.clfqp import makeproblem_clfqp

class Controller_CLF_L2_Origin(Controller): 
    def __init__(self, sys:ControlAffineSys) -> None:
        super().__init__(sys=sys)

        self.problem = makeproblem_clfqp(self)
    
    def u(self, t:float, x:np.ndarray) -> np.ndarray:
        self.u_ref_param.value = np.array([[0.]])
        clf_val,clf_grad = self.clf(x)

        self.r_penalty_param.value = np.array([[1.]])
        self.V_param.value = clf_val
        self.LfV_param.value = clf_grad@self.sys.f(0,x)
        self.LgV_param.value = clf_grad@self.sys.g(0,x)
       
        self.problem.solve()
        
        u_val = self.u_var.value
        r_val = self.r_var.value
        return u_val

    def clf(self, x:np.ndarray) -> Tuple[float,np.ndarray]:
        return np.array([[np.linalg.norm(x)**2]]) / 2 , x.T
    