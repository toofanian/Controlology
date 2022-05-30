from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp

from controllers.ctr_Parents import Controller

def makeProblem_clfqp_layer(self:Controller) -> CvxpyLayer:
        '''
        sets up the CLFQP, and returns it as a differentiable cvxpylayer
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
        objective_expression = cp.sum_squares(self.u_var - self.u_ref_param) + cp.multiply(self.r_penalty_param,self.r_var)
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
        assert problem.is_dpp(), 'Problem is not dpp'

        # convert to differentiable layer, store in self for later use
        variables = [self.u_var,self.r_var]
        parameters = [self.u_ref_param,self.r_penalty_param,self.V_param,self.LfV_param,self.LgV_param]
        problem_layer = CvxpyLayer(problem=problem,variables=variables,parameters=parameters)
        
        return problem_layer