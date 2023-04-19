from models import Simple_Pendullum
from PDP import PDPmethod
import casadi as ca

model = Simple_Pendullum(1, 4, 0.1) # m = 1, l = 1, b = 0.1
model.defineDyn()
model.define_cost_fn(terminal_cost=0.001)

OC = PDPmethod()
OC.setVariables(model.state, model.u)
OC.setAuxVar(ca.vertcat(model.dyn_auxvar, model.cost_auxvar))
