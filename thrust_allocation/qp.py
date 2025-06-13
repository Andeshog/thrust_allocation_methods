import numpy as np
import cvxpy as cp
from dataclasses import dataclass

@dataclass
class QPParams:
    dt: float = 0.1
    u_bound: float = 300.0
    max_force_rate: float = 50.0
    w_matrix: np.ndarray = 1.0 * np.eye(8)
    q_matrix: np.ndarray = 1000.0 * np.eye(3)
    beta: float = 1.1

class QPAllocator:
    def __init__(self, params: QPParams):
        """
        Initializes the QP-based thrust allocator.

        Parameters:
            half_length (float): Half the vessel length (m).
            half_width (float): Half the vessel width (m).
            u_bound (float): Thruster force bound (N); forces are in [-u_bound, u_bound].
            max_force_rate (float): Maximum allowed change in thruster force (N per timestep).
            W (np.ndarray): Weight matrix for the force components (default: identity).
            Q (np.ndarray): Weight matrix for slack variables (default: 1000*I).
            beta (float): Weight for the extra scalar variable f̄.
        """
        u_bound = params.u_bound
        max_force_rate = params.max_force_rate
        self.W = params.w_matrix
        self.Q = params.q_matrix
        self.beta = params.beta
        self.dt = params.dt
        half_length = 1.8  # Half length of the vessel (m)
        half_width = 0.8   # Half width of the vessel (m)
        self.positions = np.array([
            [ half_length, -half_width],   # front-left
            [ half_length,  half_width],   # front-right
            [-half_length, -half_width],   # rear-left
            [-half_length,  half_width]    # rear-right
        ])
        self.n_thrusters = 4

        self.alpha_des = np.array([3*np.pi/4, -3*np.pi/4, np.pi/4, -np.pi/4])
        self.alpha_prev = self.alpha_des.copy()
        self.u_prev = np.zeros(self.n_thrusters)

        self.u_bound = u_bound
        self.u_lb = np.zeros(self.n_thrusters)
        self.u_ub = u_bound * np.ones(self.n_thrusters)

        self.static_alpha_lb = np.array([np.pi/2, -np.pi, 0, -np.pi/2])
        self.static_alpha_ub = np.array([np.pi, -np.pi/2, np.pi/2, 0])

        self.max_force_rate = max_force_rate

        B_row1, B_row2, B_row3 = [], [], []
        for (x, y) in self.positions:
            B_row1.extend([1, 0])
            B_row2.extend([0, 1])
            B_row3.extend([-y, x])
        self.B = np.vstack([B_row1, B_row2, B_row3])

        self.z_dim = 2*self.n_thrusters + 3 + 1
        
        H_f = 2 * self.W
        H_s = 2 * self.Q
        H_fb = 1e-6 * np.eye(1)

        self.H_const = np.block([
            [H_f,                           np.zeros((2*self.n_thrusters, 3)),   np.zeros((2*self.n_thrusters, 1))],
            [np.zeros((3, 2*self.n_thrusters)), H_s,                         np.zeros((3, 1))],
            [np.zeros((1, 2*self.n_thrusters + 3)), H_fb]
        ])

        self.c_const = np.zeros((self.z_dim,))
        self.c_const[-1] = self.beta

    def allocate(self, tau_desired):
        """
        Solves the QP thrust allocation problem for a given desired net force/moment.

        Parameters:
            tau_desired (array_like): Desired net force/moment (3-element vector).

        Returns:
            f_ext (np.ndarray): (n_thrusters x 2) extended force vector ([Fx, Fy] per thruster).
            thrusts (np.ndarray): (n_thrusters,) thrust magnitudes.
            angles (np.ndarray): (n_thrusters,) computed azimuth angles (rad).
        """
        n = self.n_thrusters
        delta_u_max = self.max_force_rate * self.dt

        dynamic_alpha_lb = self.static_alpha_lb
        dynamic_alpha_ub = self.static_alpha_ub
        
        dynamic_u_lb = np.maximum(self.u_lb, self.u_prev - delta_u_max)
        dynamic_u_ub = np.minimum(self.u_ub, self.u_prev + delta_u_max)
        
        z = cp.Variable(self.z_dim)
        f = z[:2*n]
        s = z[2*n:2*n+3]
        f_bar = z[-1]

        eq_constr = [self.B @ f - s == tau_desired]

        constr = []
        for i in range(n):
            idx_fx = 2*i
            idx_fy = 2*i + 1
            constr.append(f[idx_fx] <= dynamic_u_ub[i])
            constr.append(f[idx_fx] >= -dynamic_u_ub[i])
            constr.append(f[idx_fy] <= dynamic_u_ub[i])
            constr.append(f[idx_fy] >= -dynamic_u_ub[i])
        
        for i in range(2*n):
            constr.append(f[i] - f_bar <= 0.0)
            constr.append(-f[i] - f_bar <= 0.0)
        constr.append(f_bar >= 1.0)
        
        epsilon = 1e-4  # small tolerance
        for i in range(n):
            Fx = f[2*i]
            Fy = f[2*i + 1]
            th_min = dynamic_alpha_lb[i]
            th_max = dynamic_alpha_ub[i]
            constr.append(np.cos(th_min) * Fy - np.sin(th_min) * Fx >= -epsilon)
            constr.append(Fx * np.sin(th_max) - np.cos(th_max) * Fy >= -epsilon)

        
        constraints = eq_constr + constr

        objective = cp.Minimize(0.5 * cp.quad_form(z, self.H_const) + self.c_const @ z)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, warm_start=True, verbose=False)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError("QP did not solve to optimality: " + prob.status)
        
        z_val = z.value
        f_val = z_val[:2*n]
        f_ext = f_val.reshape(n, 2)
        thrusts = np.linalg.norm(f_ext, axis=1)
        angles = np.arctan2(f_ext[:, 1], f_ext[:, 0])

        min_force = 1.0  # in Newtons
        for i in range(n):
            if thrusts[i] > 0 and thrusts[i] < min_force:
                angles[i] = self.alpha_des[i]
                f_ext[i, :] = np.array([np.cos(self.alpha_des[i]), np.sin(self.alpha_des[i])]) * min_force
                thrusts[i] = min_force
        
        self.alpha_prev = angles
        self.u_prev = thrusts

        tau_actual = self.B @ f_ext.reshape(8, 1)
        tau_actual = tau_actual.flatten()

        f_ext = f_ext.flatten()
        
        return f_ext, thrusts, angles, tau_actual

