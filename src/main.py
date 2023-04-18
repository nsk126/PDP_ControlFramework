import numpy as np
from numpy.linalg import norm

# TODO:
# 1. Sys Dynamics
# 2. Control Obj : Cost
# 3. Trajectories
# 4. Gradient Descent
# 5. Auxiliary control sys
# 6. Chain rule
# 7. Updare GD

# 1. Define a system. Check input & output


class SystemID:
    def __init__(self) -> None:
        self.traj = None
        self.desired_traj = None
        self.weights = None
    
    def system_dyn(self, state, action):
        # TODO: generate next state
        pass
    
    def policy_eval(self):
        
        # J = 
        pass
    
    def compute_loss(self):
        
        
        # diff between states and actions in 'traj'
        if self.traj == None or self.desired_traj == None:
            Exception("Empty Trajectories given")
        
        # TODO: func to check samples of u and x. u -> k and x -> k+1
        states, actions = self.traj
        d_states, d_actions = self.desired_traj
        
        state_diff = norm(states - d_states)**2
        act_diff = norm(actions- d_actions)**2
        
        return np.sum(state_diff) + np.sum(act_diff) # check if working
    
    def steepest_descent(self, alpha):
        
        # TODO: mod learning rate & loss gradient computation
        
        Loss_gradient = self.compute_grad()
        
        weights_new = self.weights - (alpha * Loss_gradient)
        
        self.weights = weights_new
        return weights_new
    
    def compute_grad(self):
        
        pass
        