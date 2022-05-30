from .ctr_Parents import Controller
from systems.sys_Parents import ControlAffineSys
import numpy as np
import cvxpy as cp
from typing import Tuple,Optional,Union

class Controller_CLF_DoubleIntegrator(Controller): 
    def __init__(self, sys:ControlAffineSys) -> None:
        super().__init__(sys=sys)

        self.problem = self._makeproblem_clfqp()
    
    def u(self, t:float, x:np.ndarray) -> np.ndarray:
        self.u_ref_param.value = np.array([[0.]])
        clf_val,clf_grad = self.clf(x)

        self.r_penalty_param.value = np.array([[100.]])
        self.V_param.value = clf_val
        self.LfV_param.value = clf_grad@self.sys.f(0,x)
        self.LgV_param.value = clf_grad@self.sys.g(0,x)
       
        self.problem.solve()
        
        u_val = self.u_var.value
        r_val = self.r_var.value
        return u_val

    def clf(self, x:np.ndarray) -> Tuple[float,np.ndarray]:
        clf_val = np.array([[np.linalg.norm(x)**2]])
        clf_grad = x.T
        assert clf_val.ndim == 2
        assert clf_grad.ndim == 2
        return clf_val,clf_grad
    
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
        constraints.append(self.LfV_param + self.LgV_param@self.u_var <= -self.V_param + self.r_var)

        ### assemble problem
        problem = cp.Problem(objective,constraints)
        assert problem.is_qp(), 'Problem is not qp'
        return problem