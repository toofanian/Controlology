from types import NoneType
from matplotlib import pyplot as plt
import numpy as np
from sympy import true
from utilities.utils import reftraj_lsd
import casadi as cas

from .ctr_Parents           import Controller
from systems.sys_Parents    import ControlAffineSys

class Ctr_276BPR3_CEC(Controller):
    def __init__(self, sys:ControlAffineSys) -> None:
        super().__init__(sys=sys)
        self.u_lastseq = None
        self.x_lastseq = None

    def u(self, t:float, x:np.ndarray) -> np.ndarray:    
        
        x = x - reftraj_lsd(t)
        
        class Obstacle():
            def __init__(self,position:np.ndarray,radius:float) -> None:
                self.position = position
                self.radius = radius

        obstacles = [Obstacle(np.array([[-2],[-2]]),.5),
                     Obstacle(np.array([[1],[2]]),.5)]

        ###### set up optimization problem
        prob = cas.Opti()

        # define lookahead (T_horizon = timesteps * dt )
        timesteps = 15
        dt = .5
    
        # initialize trajectory and control sequence variables
        x_var = prob.variable(self.sys.xDims,timesteps+1) # t+1 includes initial state
        u_var = prob.variable(self.sys.uDims,timesteps)
        
        ### define objective

        # weight for objective function
        Q = np.eye(2)                # position error weights
        q = np.eye(1)                # orientation error weights
        R = np.eye(self.sys.uDims)   #  control effort weights
        p = np.eye(1)*.03             # potential field weight
        
        # decay for infinite horizon formulation
        gamma = .95  # gamma < 1 since problem has no stable point

        objective_expression = 0
        potential_field = lambda x_rob,x_ob,r_ob: 1/((x_rob[0]-x_ob[0,0])**2 + (x_rob[1]-x_ob[1,0])**2)**2

        for i in range(1,timesteps+1):
            objective_expression += gamma**i * (x_var[:2,i].T @ Q @ x_var[:2,i] )       # position error
            objective_expression += gamma**i * q * (1-cas.cos(x_var[2,i]))**2           # orientation error
            objective_expression += gamma**i * (u_var[:,i-1].T @ R @ u_var[:,i-1])          # control effort
            for obstacle in obstacles:                                                  # potential field
                objective_expression += gamma**i * p * potential_field(x_var[:,i] +\
                                reftraj_lsd(t+dt*i),obstacle.position,obstacle.radius)  
        prob.minimize(objective_expression)

        # ### define constraints

        # restricted to control bounds
        for j in range(timesteps):
            for i in range(self.sys.uDims):
                prob.subject_to(u_var[i,j] >= self.sys.uBounds[i,0] ) # lower control bound 
                prob.subject_to(u_var[i,j] <= self.sys.uBounds[i,1] ) # upper control bound 

        # obeys motion model
        # TODO/BUG: self.sys.xdot() not used since casadi-numpy interface is not supported.
        # dynamics are redefined explicitly here, specifically for the sys_DiffDrive system.

        xdot = lambda t,x,u: cas.vertcat(u[0]*cas.cos(x[2] + reftraj_lsd(t)[2,0]   ),\
                                         u[0]*cas.sin(x[2] + reftraj_lsd(t)[2,0]   ),\
                                         u[1])
        rdot = lambda t: reftraj_lsd(t+dt) - reftraj_lsd(t)

        # deterministic trajectory follows (x_i,u_seq)
        for i in range(timesteps):
            x_next = x_var[:,i] + dt*xdot(t+dt*i, x_var[:,i],u_var[:,i]) - rdot(t+dt*i)
            prob.subject_to(x_var[:,i+1] == x_next)
        
        # initial conditions
        for i in range(self.sys.xDims):
            prob.subject_to(x_var[i,0] == x[i,0])

        # # avoids obstacles
        # sumsquare_2D = lambda v1_cas,v2_np: (v1_cas[0]-v2_np[0,0])**2 + (v1_cas[1]-v2_np[1,0])**2

        # class Obstacle():
        #     def __init__(self,position:np.ndarray,radius:float) -> None:
        #         self.position = position
        #         self.radius = radius

        # obstacles = [Obstacle(np.array([[-2],[-2]]),.5),
        #              Obstacle(np.array([[1],[2]]),.5)]

        # for obstacle in obstacles:
        #     for i in range(1,timesteps+1):
        #         prob.subject_to(sumsquare_2D(x_var[:,i] + reftraj_lsd(t+dt*i) ,obstacle.position) > obstacle.radius**2)
        # # TODO issue with error state determining obstacle avoidance... use ref to position yourself?
        
        
        if type(self.u_lastseq) != NoneType:
            prob.set_initial(u_var,self.u_lastseq)
        if type(self.x_lastseq) != NoneType:
            prob.set_initial(x_var,self.x_lastseq)

        prob.set_initial(x_var[:,0], x)


        opts = {'ipopt.print_level': 0, 'print_time': 0}
        prob.solver("ipopt",opts) # set numerical backend
        sol = prob.solve()
        u_optimal_step1 = np.reshape(np.array(prob.value(u_var[:,0])),(self.sys.uDims,1))

        self.u_lastseq = prob.value(u_var)
        self.x_lastseq = prob.value(x_var)

        return u_optimal_step1