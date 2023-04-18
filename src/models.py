import casadi as ca
from casadi import SX
import numpy as np

class Simple_Pendullum:
    def __init__(self, mass=None, length=None, damping_ratio=None):
        """
        Purpose: set params
        """
        self.g = 9.81 # gravity
        self.m = mass
        self.l = length
        self.damping = damping_ratio
        self.params = []
        self.cost_auxvar = []
        
    def defineDyn(self):
        
        # check if sys params are set to None. TODO: remove
        if self.m == None:
            self.m = SX.sym('m')
            self.params += [self.m]
            
        if self.l == None:
            self.l = SX.sym('l')
            self.params += [self.l]
            
        if self.damping == None:
            self.damping = SX.sym('damping_ratio')
            self.params += [self.damping]
            
        # Turn into casadi format
        self.params  = ca.vcat(self.params)
            
        # ensure float input
        self.m = float(self.m)
        self.l = float(self.l)
        self.damping = float(self.damping)
            
        # Sys dynamics
        theta = SX.sym('theta')
        theta_d = SX.sym('theta_d')
        
        # Set control input
        self.u = SX.sym('u')
        
        l = self.m * (self.l ** 2) / 3.0 # var to shorten below eq
        self.state = ca.vertcat(theta_d, (self.u - self.m * self.g * self.l * np.sin(theta) - self.damping * theta_d) / l)
        
        
    def define_cost_fn(self, stage_cost=None, terminal_cost=None, desired_state=None):

        # TODO: err check all inputs
     
        if desired_state == None:
            desired_state = np.array([np.pi, 0.0]).reshape(-1,1) # make State X in column format
            
        cost_q = self.state[0] - desired_state[0]  # angle
        cost_dq = self.state[1] - desired_state[1]  # angular velocity
        
        self.cost_q = cost_q ** 2
        self.cost_dq = cost_dq ** 2
        
        self.cost_u = ca.dot(self.u, self.u)
        
        if stage_cost == None:
            wq = SX.sym('wq')
            wq_d = SX.sym('wq_d')
            self.cost_auxvar += [wq]
            self.cost_auxvar += [wq_d]
        else:
            wq = stage_cost[0]
            wq_d = stage_cost[1]
            
        if terminal_cost == None:
            wu = SX.sym('wu')
            self.cost_auxvar += [wu]
        else:
            wu = terminal_cost
            
        # convert to casadi format
        self.cost_auxvar = ca.vcat(self.cost_auxvar)
        
        # define state costs
        self.stage_cost =   wq * self.cost_q + wq_d * self.cost_dq + wu * self.cost_u
        self.terminal_cost = wq * self.cost_q + wq_d * self.cost_dq
        
        
    def pendullum_pos(self):
        
        #
        
        pass