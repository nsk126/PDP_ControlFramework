import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import argparse
import scipy.io as sio
import PDP
import time
from models import InvPendulum

# TODO:
# 1. Sys Dynamics
# 2. Control Obj : Cost
# 3. Trajectories
# 4. Gradient Descent
# 5. Auxiliary control sys
# 6. Chain rule
# 7. Updare GD

def run_PDP(args):

    try:
        # Load Inv Pendulum System dunamics
        pendulum = InvPendulum()
        pendulum.initDyn()
        pendulum.initCost()

    except Exception as e:
        raise e

    if args.load:
        # Load demos
        print(f'demos file provided: {args.load}')

        data = sio.loadmat('data/pendulum_demos.mat')
        trajectories = data['trajectories']
        true_parameter = data['true_parameter']
        dt = data['dt']
    else:
        # Generate demos
        print('Generating demos')

        gen_demos(pendulum)


    pendulumoc = PDP.OCSys()
    pendulumoc.setAuxvarVariable(ca.vertcat(pendulum.dyn_auxvar, pendulum.cost_auxvar))
    pendulumoc.setControlVariable(pendulum.U)
    pendulumoc.setStateVariable(pendulum.X)
    dyn = pendulum.X + dt * pendulum.f
    pendulumoc.setDyn(dyn)
    pendulumoc.setPathCost(pendulum.path_cost)
    pendulumoc.setFinalCost(pendulum.final_cost)
    pendulumoc.diffPMP()
    lqr_solver = PDP.LQR()


    for j in range(1):  # trial loop
        start_time = time.time()
        lr = 1e-5  # learning rate
        # initialize
        loss_trace, parameter_trace = [], []
        sigma = 0.9
        initial_parameter = true_parameter + sigma * np.random.random(len(true_parameter)) - sigma / 2
        current_parameter = initial_parameter
        for k in range(int(1e4)):  # iteration loop (or epoch loop)
            loss = 0
            dp = np.zeros(current_parameter.shape)
            # loop for each demos trajectory
            n_demo = trajectories.shape[1]
            for i in range(n_demo):
                # demos information extraction
                demo_state_traj = trajectories[0, i]['state_traj_opt'][0, 0]
                demo_control_traj = trajectories[0, i]['control_traj_opt'][0, 0]
                demo_ini_state = demo_state_traj[0, :]
                demo_horizon = demo_control_traj.shape[0]
                # learner's current trajectory based on current parameter guess
                traj = pendulumoc.ocSolver(demo_ini_state, demo_horizon, current_parameter)
                # Establish the auxiliary control system
                aux_sys = pendulumoc.getAuxSys(state_traj_opt=traj['state_traj_opt'],
                                            control_traj_opt=traj['control_traj_opt'],
                                            costate_traj_opt=traj['costate_traj_opt'],
                                            auxvar_value=current_parameter)
                lqr_solver.setDyn(dynF=aux_sys['dynF'], dynG=aux_sys['dynG'], dynE=aux_sys['dynE'])
                lqr_solver.setPathCost(Hxx=aux_sys['Hxx'], Huu=aux_sys['Huu'], Hxu=aux_sys['Hxu'], Hux=aux_sys['Hux'],
                                    Hxe=aux_sys['Hxe'], Hue=aux_sys['Hue'])
                lqr_solver.setFinalCost(hxx=aux_sys['hxx'], hxe=aux_sys['hxe'])
                aux_sol = lqr_solver.lqrSolver(np.zeros((pendulumoc.n_state, pendulumoc.n_auxvar)), demo_horizon)
                # take solution of the auxiliary control system
                dxdp_traj = aux_sol['state_traj_opt']
                dudp_traj = aux_sol['control_traj_opt']
                # evaluate the loss
                state_traj = traj['state_traj_opt']
                control_traj = traj['control_traj_opt']
                dldx_traj = state_traj - demo_state_traj
                dldu_traj = control_traj - demo_control_traj
                loss = loss + np.linalg.norm(dldx_traj) ** 2 + np.linalg.norm(dldu_traj) ** 2
                # chain rule
                for t in range(demo_horizon):
                    dp = dp + np.matmul(dldx_traj[t, :], dxdp_traj[t]) + np.matmul(dldu_traj[t, :], dudp_traj[t])
                dp = dp + np.dot(dldx_traj[-1, :], dxdp_traj[-1])

            # take the expectation (average)
            dp = dp / n_demo
            loss = loss / n_demo
            # update
            current_parameter = current_parameter - lr * dp
            parameter_trace += [current_parameter]
            loss_trace += [loss]

            # print and terminal check
            if k % 1 == 0:
                print('trial #', j, 'iter: ', k,    ' loss: ', loss_trace[-1].tolist())

        # save
        save_data = {'trail_no': j,
                    'initial_parameter': initial_parameter,
                    'loss_trace': loss_trace,
                    'parameter_trace': parameter_trace,
                    'learning_rate': lr,
                    'time_passed': time.time() - start_time}
        sio.savemat('PDP_results_trial_' + str(j) + '.mat', {'results': save_data})


def gen_demos(pendulum):

    # gen demos
    dt = 0.1
    pendulumoc = PDP.OCSys()
    pendulumoc.setAuxvarVariable(ca.vertcat(pendulum.dyn_auxvar, pendulum.cost_auxvar))
    pendulumoc.setStateVariable(pendulum.X)
    pendulumoc.setControlVariable(pendulum.U)
    dyn = pendulum.X + dt * pendulum.f
    pendulumoc.setDyn(dyn)
    pendulumoc.setPathCost(pendulum.path_cost)
    pendulumoc.setFinalCost(pendulum.final_cost)
    lqr_solver = PDP.LQR()


    # Save mat file
    true_parameter = [1, 1, 0.1, 10, 1]
    horizon = 20
    demos = []
    ini_state = np.zeros(pendulumoc.n_state)
    ini_q = [0, -1, -0.5, 0.5, 1]
    for i in range(5):  # generate 5 dmonstrations with each with different initial q
        print(f'Demo #{i}')
        ini_state[0] = ini_q[i]
        sol = pendulumoc.ocSolver(ini_state=ini_state, horizon=horizon, auxvar_value=true_parameter)
        pendulum.play_animation(len=1, dt=dt, state_traj=sol['state_traj_opt'])
        demos += [sol]
    # save
    sio.savemat('data/pendulum_demos.mat', {'trajectories': demos,
                                            'dt': dt,
                                            'true_parameter': true_parameter})



def main():
    parser = argparse.ArgumentParser(description='Our custom PDP program')
    parser.add_argument('-l', '--load', type=str, help='load a demos file')

    args = parser.parse_args()

    run_PDP(args)

if __name__ == '__main__':
    main()