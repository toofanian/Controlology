
import numpy as np
import cvxpy as cp

from typing import Tuple,Optional,Union
from numpy import ndarray

from .ctr_Parents           import Controller
from systems.sys_Parents    import ControlAffineSys

class Controller_CLF_L2_Origin(Controller): 
    '''
    Assumes V(x) = (1/2)*||x||_2^2 is a valid lyapunov function, then
    solves a CLFQP to return a goal-reaching control. If this assumption is false, control behavior is unqualified.

    r_penalty_param may need adjusting, depending on control and state magnitudes.
    '''

    def __init__(self, sys: ControlAffineSys, ref: Optional['Controller'] = None) -> None:
        super().__init__(sys=sys, ref=ref)
        
        ###### set up the CLFQP problem using cvxpy

        ### define variables and parameters
        self.u_var            = cp.Variable((self.sys.uDims,1))
        self.u_ref_param      = cp.Parameter((self.sys.uDims,1))
        self.r_var            = cp.Variable((1,1),nonneg=True)
        self.r_penalty_param  = cp.Parameter((1,1),nonneg=True)
        self.V_param          = cp.Parameter((1,1),nonneg=True) 
        self.LfV_param        = cp.Parameter((1,1))
        self.LgV_param        = cp.Parameter((1,self.sys.uDims))
        self.c = 1

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
        clf_val,clf_grad = self._clf(x)

        # assign clfqp parameter values
        self.u_ref_param.value      = self.ref.u(x,t) if type(self.ref) == Controller else np.zeros((self.sys.uDims,1))
        self.r_penalty_param.value  = np.array([[1.]]) # TODO make dynamic?
        self.V_param.value          = clf_val
        self.LfV_param.value        = clf_grad@self.sys.f(x,t)
        self.LgV_param.value        = clf_grad@self.sys.g(x,t)
       
        self.problem.solve(solver=cp.ECOS)
        
        u_val = self.u_var.value
        r_val = self.r_var.value
        return u_val

    def _clf(self, x:np.ndarray) -> Tuple[float,np.ndarray]:
        return np.array([[np.linalg.norm(x)**2]]) / 2 , x.T
    