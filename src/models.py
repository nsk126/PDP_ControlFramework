from casadi import *
import math
from matplotlib import pyplot as plt
import matplotlib.animation as animation

# inverted pendulum
class InvPendulum:
    def __init__(self, project_name='single pendlumn system'):
        self.project_name = project_name

    def initDyn(self, l=None, m=None, damping_ratio=None):
        # set parameter
        g = 10

        # declare system parameter
        parameter = []
        if l is None:
            self.l = SX.sym('l')
            parameter += [self.l]
        else:
            self.l = l

        if m is None:
            self.m = SX.sym('m')
            parameter += [self.m]
        else:
            self.m = m

        if damping_ratio is None:
            self.damping_ratio = SX.sym('damping_ratio')
            parameter += [self.damping_ratio]
        else:
            self.damping_ratio = damping_ratio

        self.dyn_auxvar = vcat(parameter)

        # set variable
        self.q, self.dq = SX.sym('q'), SX.sym('dq')
        self.X = vertcat(self.q, self.dq)
        U = SX.sym('u')
        self.U = U
        I = 1 / 3 * self.m * self.l * self.l
        self.f = vertcat(self.dq,
                         (self.U - self.m * g * self.l * sin(
                             self.q) - self.damping_ratio * self.dq) / I)  # continuous state-space representation

    def initCost(self, wq=None, wdq=None, wu=0.001):
        parameter = []
        if wq is None:
            self.wq = SX.sym('wq')
            parameter += [self.wq]
        else:
            self.wq = wq

        if wdq is None:
            self.wdq = SX.sym('wdq')
            parameter += [self.wdq]
        else:
            self.wdq = wdq

        self.cost_auxvar = vcat(parameter)

        # control goal
        x_goal = [math.pi, 0, 0, 0]

        # cost for q
        self.cost_q = (self.q - x_goal[0]) ** 2
        # cost for dq
        self.cost_dq = (self.dq - x_goal[1]) ** 2
        # cost for u
        self.cost_u = dot(self.U, self.U)

        self.path_cost = self.wq * self.cost_q + self.wdq * self.cost_dq + wu * self.cost_u
        self.final_cost = self.wq * self.cost_q + self.wdq * self.cost_dq

    def get_pendulum_position(self, len, state_traj):

        position = np.zeros((state_traj.shape[0], 2))
        for t in range(state_traj.shape[0]):
            q = state_traj[t, 0]
            pos_x = len * sin(q)
            pos_y = -len * cos(q)
            position[t, :] = np.array([pos_x, pos_y])
        return position

    def play_animation(self, len, dt, state_traj, state_traj_ref=None, save_option=0):

        # get the position of cart pole
        position = self.get_pendulum_position(len, state_traj)
        horizon = position.shape[0]
        if state_traj_ref is not None:
            position_ref = self.get_pendulum_position(len, state_traj_ref)
        else:
            position_ref = np.zeros_like(position)
        assert position.shape[0] == position_ref.shape[0], 'reference trajectory should have the same length'

        # set figure
        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-4, 4), ylim=(-4, 4), )
        ax.set_aspect('equal')
        ax.grid()
        ax.set_ylabel('Vertical (m)')
        ax.set_xlabel('Horizontal (m)')
        ax.set_title('Pendulum system')
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        # set lines
        cart_h, cart_w = 0.5, 1
        line, = ax.plot([], [], 'o-', lw=2)
        line_ref, = ax.plot([], [], color='lightgray', marker='o', lw=1)

        def init():
            line.set_data([], [])
            line_ref.set_data([], [])
            time_text.set_text('')
            return line, line_ref, time_text

        def animate(i):
            seg_x = [0, position[i, 0]]
            seg_y = [0, position[i, 1]]
            line.set_data(seg_x, seg_y)

            seg_x_ref = [0, position_ref[i, 0]]
            seg_y_ref = [0, position_ref[i, 1]]
            line_ref.set_data(seg_x_ref, seg_y_ref)

            time_text.set_text(time_template % (i * dt))

            return line, line_ref, time_text

        ani = animation.FuncAnimation(fig, animate, np.size(state_traj, 0),
                                      interval=50, init_func=init)

        if save_option != 0:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=-1)
            ani.save('Pendulum.mp4', writer=writer)
            print('save_success')

        plt.show()
