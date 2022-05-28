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
np.random.seed(123)
import matplotlib.pyplot as plt
from  scipy.integrate import solve_ivp

# Standard Python
import sys
from typing import Optional, Tuple, Union

# Custom Classes
from train import NN_clf, InvertedPendulum, makeProblem_CLFQP


# class Controller():
#     def __init__(self) -> None:
#         pass

# class PD_controller(Controller):
#     def __init__(self) -> None:
#         super().__init__()




# class Certificate():
#     def __init__(self) -> None:
#         pass

# class nCLF(Certificate):
#     def __init__(self) -> None:
#         super().__init__()





# class Test_Control():

#     def __init__(self) -> None:
#         self.controller:Controller
#         self.certificate:Certificate
        


#     def set_controller(self,controller:Controller):
#         self.controller=controller

#     def set_certificate(self,certificate:Certificate):
#         self.certificate=certificate






class Test_ControlCert():

    def __init__(self,
                 dynsys:InvertedPendulum,
                 clf_path:Optional[str]=None,
                 cbf_path:Optional[str]=None
                 ) -> None:
                 
        self.dynsys:InvertedPendulum = dynsys()

        if type(clf_path) == str: self.trainedCLF:nn.Module = torch.load(clf_path)
        if type(cbf_path) == str: self.trainedCBF:nn.Module = torch.load(cbf_path)

        self.problem = makeProblem_CLFQP()

    def control_PD(self,x:Union[torch.Tensor,np.ndarray]):
        '''
        Linear feedback PD controller
        '''
        kp = 100 # 100 nominal
        kd = -1  # 20 nominal
        Kgain = np.array([[kp,kd]])
        return -(x @ Kgain.T)

    def control_nclf(self,x:Union[torch.Tensor,np.ndarray]):
        '''
        uses neural clf to filter out controls based on some reference
        '''
        if type(x) == np.ndarray: x = torch.tensor(x)
        f,g = self.dynsys.getDynamics(x)

        x.requires_grad_(True)
        clf_val = self.trainedCLF(x.float())
        clf_val.backward()
        Lf_V = (x.grad@f.T).type_as(clf_val)
        Lg_V = (x.grad@g.T).type_as(clf_val)
        x.requires_grad_(False)

        # send V, Lf_V, Lg_V to clfqp, solve for r (must be done one at a time, not as batch)
        u_ref = torch.tensor(self.control_PD(x.detach().numpy()),dtype=torch.float32)
        relax_penalty = torch.tensor([100.],dtype=torch.float32)
        params = [u_ref,relax_penalty,clf_val,Lf_V,Lg_V]
        u,r = self.problem(*params,solver_args={"max_iters": 50000000})

        return u.detach().numpy()


    def test(self,controlfunc):
        def odefunc(t,x):
            f,g = self.dynsys.getDynamics(x)
            u = controlfunc(x)
            dxdt = f+g*u
            #print(u)
            return dxdt

        xi = np.array([2,2])
        tspan = (0,10)
        sol = solve_ivp(odefunc,tspan,xi)
        plt.plot(sol.t,sol.y[0,:])
        plt.plot(sol.t,sol.y[1,:])
        plt.show()
        



if __name__ == '__main__':
    tester = Test_ControlCert(InvertedPendulum,'trainedCLF.pth')
    tester.test(tester.control_nclf)
