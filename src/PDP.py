from casadi import *
import numpy
from scipy import interpolate
import casadi

class OCSys:

    def __init__(self, project_name="my optimal control system"):
        self.project_name = project_name

    def setAuxvarVariable(self, auxvar=None):
        if auxvar is None or auxvar.numel() == 0:
            self.auxvar = SX.sym('auxvar')
        else:
            self.auxvar = auxvar
        self.n_auxvar = self.auxvar.numel()

    def setStateVariable(self, state, state_lb=[], state_ub=[]):
        self.state = state
        self.n_state = self.state.numel()
        if len(state_lb) == self.n_state:
            self.state_lb = state_lb
        else:
            self.state_lb = self.n_state * [-1e20]

        if len(state_ub) == self.n_state:
            self.state_ub = state_ub
        else:
            self.state_ub = self.n_state * [1e20]

    def setControlVariable(self, control, control_lb=[], control_ub=[]):
        self.control = control
        self.n_control = self.control.numel()

        if len(control_lb) == self.n_control:
            self.control_lb = control_lb
        else:
            self.control_lb = self.n_control * [-1e20]

        if len(control_ub) == self.n_control:
            self.control_ub = control_ub
        else:
            self.control_ub = self.n_control * [1e20]

    def setDyn(self, ode):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        self.dyn = ode
        self.dyn_fn = casadi.Function(
            'dynamics', [self.state, self.control, self.auxvar], [self.dyn])

    def setPathCost(self, path_cost):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        assert path_cost.numel() == 1, "path_cost must be a scalar function"

        self.path_cost = path_cost
        self.path_cost_fn = casadi.Function(
            'path_cost', [self.state, self.control, self.auxvar], [self.path_cost])

    def setFinalCost(self, final_cost):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        assert final_cost.numel() == 1, "final_cost must be a scalar function"

        self.final_cost = final_cost
        self.final_cost_fn = casadi.Function(
            'final_cost', [self.state, self.auxvar], [self.final_cost])

    def ocSolver(self, ini_state, horizon, auxvar_value=1, print_level=0, costate_option=0):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'control'), "Define the control variable first!"
        assert hasattr(self, 'dyn'), "Define the system dynamics first!"
        assert hasattr(
            self, 'path_cost'), "Define the running cost function first!"
        assert hasattr(
            self, 'final_cost'), "Define the final cost function first!"

        if type(ini_state) == numpy.ndarray:
            ini_state = ini_state.flatten().tolist()

        # Start with an empty NLP
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # "Lift" initial conditions
        Xk = MX.sym('X0', self.n_state)
        w += [Xk]
        lbw += ini_state
        ubw += ini_state
        w0 += ini_state

        # Formulate the NLP
        for k in range(horizon):
            # New NLP variable for the control
            Uk = MX.sym('U_' + str(k), self.n_control)
            w += [Uk]
            lbw += self.control_lb
            ubw += self.control_ub
            w0 += [0.5 * (x + y)
                   for x, y in zip(self.control_lb, self.control_ub)]

            # Integrate till the end of the interval
            Xnext = self.dyn_fn(Xk, Uk, auxvar_value)
            Ck = self.path_cost_fn(Xk, Uk, auxvar_value)
            J = J + Ck

            # New NLP variable for state at end of interval
            Xk = MX.sym('X_' + str(k + 1), self.n_state)
            w += [Xk]
            lbw += self.state_lb
            ubw += self.state_ub
            w0 += [0.5 * (x + y) for x, y in zip(self.state_lb, self.state_ub)]

            # Add equality constraint
            g += [Xnext - Xk]
            lbg += self.n_state * [0]
            ubg += self.n_state * [0]

        # Adding the final cost
        J = J + self.final_cost_fn(Xk, auxvar_value)

        # Create an NLP solver and solve it
        opts = {'ipopt.print_level': print_level,
                'ipopt.sb': 'yes', 'print_time': print_level}
        prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
        solver = nlpsol('solver', 'ipopt', prob, opts)
        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()

        # take the optimal control and state
        sol_traj = numpy.concatenate((w_opt, self.n_control * [0]))
        sol_traj = numpy.reshape(sol_traj, (-1, self.n_state + self.n_control))
        state_traj_opt = sol_traj[:, 0:self.n_state]
        control_traj_opt = numpy.delete(sol_traj[:, self.n_state:], -1, 0)
        time = numpy.array([k for k in range(horizon + 1)])

        # Compute the costates using two options
        if costate_option == 0:
            # Default option, which directly obtains the costates from the NLP solver
            costate_traj_opt = numpy.reshape(
                sol['lam_g'].full().flatten(), (-1, self.n_state))
        else:
            # Another option, which solve the costates by the Pontryagin's Maximum Principle
            # The variable name is consistent with the notations used in the PDP paper
            dfx_fun = casadi.Function('dfx', [self.state, self.control, self.auxvar], [
                                      jacobian(self.dyn, self.state)])
            dhx_fun = casadi.Function('dhx', [self.state, self.auxvar], [
                                      jacobian(self.final_cost, self.state)])
            dcx_fun = casadi.Function('dcx', [self.state, self.control, self.auxvar],
                                      [jacobian(self.path_cost, self.state)])
            costate_traj_opt = numpy.zeros((horizon, self.n_state))
            costate_traj_opt[-1,
                             :] = dhx_fun(state_traj_opt[-1, :], auxvar_value)
            for k in range(horizon - 1, 0, -1):
                costate_traj_opt[k - 1, :] = dcx_fun(state_traj_opt[k, :], control_traj_opt[k, :],
                                                     auxvar_value).full() + numpy.dot(
                    numpy.transpose(
                        dfx_fun(state_traj_opt[k, :], control_traj_opt[k, :], auxvar_value).full()),
                    costate_traj_opt[k, :])

        # output
        opt_sol = {"state_traj_opt": state_traj_opt,
                   "control_traj_opt": control_traj_opt,
                   "costate_traj_opt": costate_traj_opt,
                   'auxvar_value': auxvar_value,
                   "time": time,
                   "horizon": horizon,
                   "cost": sol['f'].full()}

        return opt_sol

    def diffPMP(self):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'control'), "Define the control variable first!"
        assert hasattr(self, 'dyn'), "Define the system dynamics first!"
        assert hasattr(
            self, 'path_cost'), "Define the running cost/reward function first!"
        assert hasattr(
            self, 'final_cost'), "Define the final cost/reward function first!"

        # Define the Hamiltonian function
        self.costate = casadi.SX.sym('lambda', self.state.numel())
        self.path_Hamil = self.path_cost + \
            dot(self.dyn, self.costate)  # path Hamiltonian
        self.final_Hamil = self.final_cost  # final Hamiltonian

        # Differentiating dynamics; notations here are consistent with the PDP paper
        self.dfx = jacobian(self.dyn, self.state)
        self.dfx_fn = casadi.Function(
            'dfx', [self.state, self.control, self.auxvar], [self.dfx])
        self.dfu = jacobian(self.dyn, self.control)
        self.dfu_fn = casadi.Function(
            'dfu', [self.state, self.control, self.auxvar], [self.dfu])
        self.dfe = jacobian(self.dyn, self.auxvar)
        self.dfe_fn = casadi.Function(
            'dfe', [self.state, self.control, self.auxvar], [self.dfe])

        # First-order derivative of path Hamiltonian
        self.dHx = jacobian(self.path_Hamil, self.state).T
        self.dHx_fn = casadi.Function(
            'dHx', [self.state, self.control, self.costate, self.auxvar], [self.dHx])
        self.dHu = jacobian(self.path_Hamil, self.control).T
        self.dHu_fn = casadi.Function(
            'dHu', [self.state, self.control, self.costate, self.auxvar], [self.dHu])

        # Second-order derivative of path Hamiltonian
        self.ddHxx = jacobian(self.dHx, self.state)
        self.ddHxx_fn = casadi.Function(
            'ddHxx', [self.state, self.control, self.costate, self.auxvar], [self.ddHxx])
        self.ddHxu = jacobian(self.dHx, self.control)
        self.ddHxu_fn = casadi.Function(
            'ddHxu', [self.state, self.control, self.costate, self.auxvar], [self.ddHxu])
        self.ddHxe = jacobian(self.dHx, self.auxvar)
        self.ddHxe_fn = casadi.Function(
            'ddHxe', [self.state, self.control, self.costate, self.auxvar], [self.ddHxe])
        self.ddHux = jacobian(self.dHu, self.state)
        self.ddHux_fn = casadi.Function(
            'ddHux', [self.state, self.control, self.costate, self.auxvar], [self.ddHux])
        self.ddHuu = jacobian(self.dHu, self.control)
        self.ddHuu_fn = casadi.Function(
            'ddHuu', [self.state, self.control, self.costate, self.auxvar], [self.ddHuu])
        self.ddHue = jacobian(self.dHu, self.auxvar)
        self.ddHue_fn = casadi.Function(
            'ddHue', [self.state, self.control, self.costate, self.auxvar], [self.ddHue])

        # First-order derivative of final Hamiltonian
        self.dhx = jacobian(self.final_Hamil, self.state).T
        self.dhx_fn = casadi.Function(
            'dhx', [self.state, self.auxvar], [self.dhx])

        # second order differential of path Hamiltonian
        self.ddhxx = jacobian(self.dhx, self.state)
        self.ddhxx_fn = casadi.Function(
            'ddhxx', [self.state, self.auxvar], [self.ddhxx])
        self.ddhxe = jacobian(self.dhx, self.auxvar)
        self.ddhxe_fn = casadi.Function(
            'ddhxe', [self.state, self.auxvar], [self.ddhxe])

    def getAuxSys(self, state_traj_opt, control_traj_opt, costate_traj_opt, auxvar_value=1):
        statement = [hasattr(self, 'dfx_fn'), hasattr(self, 'dfu_fn'), hasattr(self, 'dfe_fn'),
                     hasattr(self, 'ddHxx_fn'),
                     hasattr(self, 'ddHxu_fn'), hasattr(
                         self, 'ddHxe_fn'), hasattr(self, 'ddHux_fn'),
                     hasattr(self, 'ddHuu_fn'),
                     hasattr(self, 'ddHue_fn'), hasattr(self, 'ddhxx_fn'), hasattr(self, 'ddhxe_fn'), ]
        if not all(statement):
            self.diffPMP()

        # Initialize the coefficient matrices of the auxiliary control system: note that all the notations used here are
        # consistent with the notations defined in the PDP paper.
        dynF, dynG, dynE = [], [], []
        matHxx, matHxu, matHxe, matHux, matHuu, matHue, mathxx, mathxe = [
        ], [], [], [], [], [], [], []

        # Solve the above coefficient matrices
        for t in range(numpy.size(control_traj_opt, 0)):
            curr_x = state_traj_opt[t, :]
            curr_u = control_traj_opt[t, :]
            next_lambda = costate_traj_opt[t, :]
            dynF += [self.dfx_fn(curr_x, curr_u, auxvar_value).full()]
            dynG += [self.dfu_fn(curr_x, curr_u, auxvar_value).full()]
            dynE += [self.dfe_fn(curr_x, curr_u, auxvar_value).full()]
            matHxx += [self.ddHxx_fn(curr_x, curr_u,
                                     next_lambda, auxvar_value).full()]
            matHxu += [self.ddHxu_fn(curr_x, curr_u,
                                     next_lambda, auxvar_value).full()]
            matHxe += [self.ddHxe_fn(curr_x, curr_u,
                                     next_lambda, auxvar_value).full()]
            matHux += [self.ddHux_fn(curr_x, curr_u,
                                     next_lambda, auxvar_value).full()]
            matHuu += [self.ddHuu_fn(curr_x, curr_u,
                                     next_lambda, auxvar_value).full()]
            matHue += [self.ddHue_fn(curr_x, curr_u,
                                     next_lambda, auxvar_value).full()]
        mathxx = [self.ddhxx_fn(state_traj_opt[-1, :], auxvar_value).full()]
        mathxe = [self.ddhxe_fn(state_traj_opt[-1, :], auxvar_value).full()]

        auxSys = {"dynF": dynF,
                  "dynG": dynG,
                  "dynE": dynE,
                  "Hxx": matHxx,
                  "Hxu": matHxu,
                  "Hxe": matHxe,
                  "Hux": matHux,
                  "Huu": matHuu,
                  "Hue": matHue,
                  "hxx": mathxx,
                  "hxe": mathxe}
        return auxSys


class LQR:

    def __init__(self, project_name="LQR system"):
        self.project_name = project_name

    def setDyn(self, dynF, dynG, dynE=None):
        if type(dynF) is numpy.ndarray:
            self.dynF = [dynF]
            self.n_state = numpy.size(dynF, 0)
        elif type(dynF[0]) is numpy.ndarray:
            self.dynF = dynF
            self.n_state = numpy.size(dynF[0], 0)
        else:
            assert False, "Type of dynF matrix should be numpy.ndarray  or list of numpy.ndarray"

        if type(dynG) is numpy.ndarray:
            self.dynG = [dynG]
            self.n_control = numpy.size(dynG, 1)
        elif type(dynG[0]) is numpy.ndarray:
            self.dynG = dynG
            self.n_control = numpy.size(self.dynG[0], 1)
        else:
            assert False, "Type of dynG matrix should be numpy.ndarray  or list of numpy.ndarray"

        if dynE is not None:
            if type(dynE) is numpy.ndarray:
                self.dynE = [dynE]
                self.n_batch = numpy.size(dynE, 1)
            elif type(dynE[0]) is numpy.ndarray:
                self.dynE = dynE
                self.n_batch = numpy.size(dynE[0], 1)
            else:
                assert False, "Type of dynE matrix should be numpy.ndarray, list of numpy.ndarray, or None"
        else:
            self.dynE = None
            self.n_batch = None

    def setPathCost(self, Hxx, Huu, Hxu=None, Hux=None, Hxe=None, Hue=None):

        if type(Hxx) is numpy.ndarray:
            self.Hxx = [Hxx]
        elif type(Hxx[0]) is numpy.ndarray:
            self.Hxx = Hxx
        else:
            assert False, "Type of path cost Hxx matrix should be numpy.ndarray or list of numpy.ndarray, or None"

        if type(Huu) is numpy.ndarray:
            self.Huu = [Huu]
        elif type(Huu[0]) is numpy.ndarray:
            self.Huu = Huu
        else:
            assert False, "Type of path cost Huu matrix should be numpy.ndarray or list of numpy.ndarray, or None"

        if Hxu is not None:
            if type(Hxu) is numpy.ndarray:
                self.Hxu = [Hxu]
            elif type(Hxu[0]) is numpy.ndarray:
                self.Hxu = Hxu
            else:
                assert False, "Type of path cost Hxu matrix should be numpy.ndarray or list of numpy.ndarray, or None"
        else:
            self.Hxu = None

        if Hux is not None:
            if type(Hux) is numpy.ndarray:
                self.Hux = [Hux]
            elif type(Hux[0]) is numpy.ndarray:
                self.Hux = Hux
            else:
                assert False, "Type of path cost Hux matrix should be numpy.ndarray or list of numpy.ndarray, or None"
        else:
            self.Hux = None

        if Hxe is not None:
            if type(Hxe) is numpy.ndarray:
                self.Hxe = [Hxe]
            elif type(Hxe[0]) is numpy.ndarray:
                self.Hxe = Hxe
            else:
                assert False, "Type of path cost Hxe matrix should be numpy.ndarray or list of numpy.ndarray, or None"
        else:
            self.Hxe = None

        if Hue is not None:
            if type(Hue) is numpy.ndarray:
                self.Hue = [Hue]
            elif type(Hue[0]) is numpy.ndarray:
                self.Hue = Hue
            else:
                assert False, "Type of path cost Hue matrix should be numpy.ndarray or list of numpy.ndarray, or None"
        else:
            self.Hue = None

    def setFinalCost(self, hxx, hxe=None):

        if type(hxx) is numpy.ndarray:
            self.hxx = [hxx]
        elif type(hxx[0]) is numpy.ndarray:
            self.hxx = hxx
        else:
            assert False, "Type of final cost hxx matrix should be numpy.ndarray or list of numpy.ndarray"

        if hxe is not None:
            if type(hxe) is numpy.ndarray:
                self.hxe = [hxe]
            elif type(hxe[0]) is numpy.ndarray:
                self.hxe = hxe
            else:
                assert False, "Type of final cost hxe matrix should be numpy.ndarray, list of numpy.ndarray, or None"
        else:
            self.hxe = None

    def lqrSolver(self, ini_state, horizon):

        # Data pre-processing
        n_state = numpy.size(self.dynF[0], 1)
        if type(ini_state) is list:
            self.ini_x = numpy.array(ini_state, numpy.float64)
            if self.ini_x.ndim == 2:
                self.n_batch = numpy.size(self.ini_x, 1)
            else:
                self.n_batch = 1
                self.ini_x = self.ini_x.reshape(n_state, -1)
        elif type(ini_state) is numpy.ndarray:
            self.ini_x = ini_state
            if self.ini_x.ndim == 2:
                self.n_batch = numpy.size(self.ini_x, 1)
            else:
                self.n_batch = 1
                self.ini_x = self.ini_x.reshape(n_state, -1)
        else:
            assert False, "Initial state should be of numpy.ndarray type or list!"

        self.horizon = horizon

        if self.dynE is not None:
            assert self.n_batch == numpy.size(self.dynE[0],
                                              1), "Number of data batch is not consistent with column of dynE"

        # Check the time horizon
        if len(self.dynF) > 1 and len(self.dynF) != self.horizon:
            assert False, "time-varying dynF is not consistent with given horizon"
        elif len(self.dynF) == 1:
            F = self.horizon * self.dynF
        else:
            F = self.dynF

        if len(self.dynG) > 1 and len(self.dynG) != self.horizon:
            assert False, "time-varying dynG is not consistent with given horizon"
        elif len(self.dynG) == 1:
            G = self.horizon * self.dynG
        else:
            G = self.dynG

        if self.dynE is not None:
            if len(self.dynE) > 1 and len(self.dynE) != self.horizon:
                assert False, "time-varying dynE is not consistent with given horizon"
            elif len(self.dynE) == 1:
                E = self.horizon * self.dynE
            else:
                E = self.dynE
        else:
            E = self.horizon * [numpy.zeros(self.ini_x.shape)]

        if len(self.Hxx) > 1 and len(self.Hxx) != self.horizon:
            assert False, "time-varying Hxx is not consistent with given horizon"
        elif len(self.Hxx) == 1:
            Hxx = self.horizon * self.Hxx
        else:
            Hxx = self.Hxx

        if len(self.Huu) > 1 and len(self.Huu) != self.horizon:
            assert False, "time-varying Huu is not consistent with given horizon"
        elif len(self.Huu) == 1:
            Huu = self.horizon * self.Huu
        else:
            Huu = self.Huu

        hxx = self.hxx

        if self.hxe is None:
            hxe = [numpy.zeros(self.ini_x.shape)]

        if self.Hxu is None:
            Hxu = self.horizon * [numpy.zeros((self.n_state, self.n_control))]
        else:
            if len(self.Hxu) > 1 and len(self.Hxu) != self.horizon:
                assert False, "time-varying Hxu is not consistent with given horizon"
            elif len(self.Hxu) == 1:
                Hxu = self.horizon * self.Hxu
            else:
                Hxu = self.Hxu

        if self.Hux is None:  # Hux is the transpose of Hxu
            Hux = self.horizon * [numpy.zeros((self.n_control, self.n_state))]
        else:
            if len(self.Hux) > 1 and len(self.Hux) != self.horizon:
                assert False, "time-varying Hux is not consistent with given horizon"
            elif len(self.Hux) == 1:
                Hux = self.horizon * self.Hux
            else:
                Hux = self.Hux

        if self.Hxe is None:
            Hxe = self.horizon * [numpy.zeros((self.n_state, self.n_batch))]
        else:
            if len(self.Hxe) > 1 and len(self.Hxe) != self.horizon:
                assert False, "time-varying Hxe is not consistent with given horizon"
            elif len(self.Hxe) == 1:
                Hxe = self.horizon * self.Hxe
            else:
                Hxe = self.Hxe

        if self.Hue is None:
            Hue = self.horizon * [numpy.zeros((self.n_control, self.n_batch))]
        else:
            if len(self.Hue) > 1 and len(self.Hue) != self.horizon:
                assert False, "time-varying Hue is not consistent with given horizon"
            elif len(self.Hue) == 1:
                Hue = self.horizon * self.Hue
            else:
                Hue = self.Hue

        # Solve the Riccati equations: the notations used here are consistent with Lemma 4.2 in the PDP paper
        I = numpy.eye(self.n_state)
        PP = self.horizon * [numpy.zeros((self.n_state, self.n_state))]
        WW = self.horizon * [numpy.zeros((self.n_state, self.n_batch))]
        PP[-1] = self.hxx[0]
        WW[-1] = self.hxe[0]
        for t in range(self.horizon - 1, 0, -1):
            P_next = PP[t]
            W_next = WW[t]
            invHuu = numpy.linalg.inv(Huu[t])
            GinvHuu = numpy.matmul(G[t], invHuu)
            HxuinvHuu = numpy.matmul(Hxu[t], invHuu)
            A_t = F[t] - numpy.matmul(GinvHuu, numpy.transpose(Hxu[t]))
            R_t = numpy.matmul(GinvHuu, numpy.transpose(G[t]))
            M_t = E[t] - numpy.matmul(GinvHuu, Hue[t])
            Q_t = Hxx[t] - numpy.matmul(HxuinvHuu, numpy.transpose(Hxu[t]))
            N_t = Hxe[t] - numpy.matmul(HxuinvHuu, Hue[t])

            temp_mat = numpy.matmul(numpy.transpose(
                A_t), numpy.linalg.inv(I + numpy.matmul(P_next, R_t)))
            P_curr = Q_t + numpy.matmul(temp_mat, numpy.matmul(P_next, A_t))
            W_curr = N_t + numpy.matmul(temp_mat,
                                        W_next + numpy.matmul(P_next, M_t))

            PP[t - 1] = P_curr
            WW[t - 1] = W_curr

        # Compute the trajectory using the Raccti matrices obtained from the above: the notations used here are
        # consistent with the PDP paper in Lemma 4.2
        state_traj_opt = (self.horizon + 1) * \
            [numpy.zeros((self.n_state, self.n_batch))]
        control_traj_opt = (self.horizon) * \
            [numpy.zeros((self.n_control, self.n_batch))]
        costate_traj_opt = (self.horizon) * \
            [numpy.zeros((self.n_state, self.n_batch))]
        state_traj_opt[0] = self.ini_x
        for t in range(self.horizon):
            P_next = PP[t]
            W_next = WW[t]
            invHuu = numpy.linalg.inv(Huu[t])
            GinvHuu = numpy.matmul(G[t], invHuu)
            A_t = F[t] - numpy.matmul(GinvHuu, numpy.transpose(Hxu[t]))
            M_t = E[t] - numpy.matmul(GinvHuu, Hue[t])
            R_t = numpy.matmul(GinvHuu, numpy.transpose(G[t]))

            x_t = state_traj_opt[t]
            u_t = -numpy.matmul(invHuu, numpy.matmul(numpy.transpose(Hxu[t]), x_t) + Hue[t]) \
                  - numpy.linalg.multi_dot([invHuu, numpy.transpose(G[t]), numpy.linalg.inv(I + numpy.dot(P_next, R_t)),
                                            (numpy.matmul(numpy.matmul(P_next, A_t), x_t) + numpy.matmul(P_next,
                                                                                                         M_t) + W_next)])

            x_next = numpy.matmul(F[t], x_t) + numpy.matmul(G[t], u_t) + E[t]
            lambda_next = numpy.matmul(P_next, x_next) + W_next

            state_traj_opt[t + 1] = x_next
            control_traj_opt[t] = u_t
            costate_traj_opt[t] = lambda_next
        time = [k for k in range(self.horizon + 1)]

        opt_sol = {'state_traj_opt': state_traj_opt,
                   'control_traj_opt': control_traj_opt,
                   'costate_traj_opt': costate_traj_opt,
                   'time': time}
        return opt_sol
