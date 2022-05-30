import torch
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
import numpy as np






u_var = cp.Variable((1,1))
r_var = cp.Variable((1,1),nonneg=True)

lower_param = cp.Parameter((1,1))
upper_param = cp.Parameter((1,1))

objective_expression = (u_var**2) + 200000*r_var
objective = cp.Minimize(objective_expression)

### define constraints
constraints = []
# control constraints
constraints.append(u_var>=lower_param)
constraints.append(u_var<=upper_param)

constraints.append(.01*u_var-r_var <= -100.)

### assemble problem
problem = cp.Problem(objective,constraints)
assert problem.is_dpp(), 'Problem is not dpp'

# convert to differentiable layer, store in self for later use
variables = [u_var,r_var]
parameters = [lower_param,upper_param]
problem_layer = CvxpyLayer(problem=problem,variables=variables,parameters=parameters)


lower_ten = torch.tensor([[-10000.]])
upper_ten = torch.tensor([[10000.]])
params = [lower_ten,upper_ten]
u,r = problem_layer(*params)
print(u,r)


