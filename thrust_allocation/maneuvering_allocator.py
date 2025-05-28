import numpy as np
from scipy.linalg import null_space

class ManeuveringAllocator:
    def __init__(self):
        self.power_management = False

        self.gamma = 0.1
        self.mu = 10.0
        self.w_matrix = np.eye(8)
        self.last_xi_p = np.zeros(8)
        self.xi = np.zeros(8)
        self.theta = np.zeros(5)
        self.rho = 0.1
        self.zeta = 0.0001
        self.rate_limit = 100.0
        self.c = np.ones(4)
        self.lambda_ = 1.0

        self.theta_min = -50.0
        self.theta_max = 50.0

        self.min_forces = np.array([-300.0, 1.0, -300.0, -300.0, 1.0, 1.0, 1.0, -300.0])
        self.max_forces = np.array([-1.0, 300.0, -1.0, -1.0, 300.0, 300.0, 300.0, -1.0])
        self.reference_angles = np.array([3 * np.pi / 4, -3 * np.pi / 4, np.pi/4, -np.pi/4])

        self.num_thrusters = 4

        half_length = 1.8
        half_width = 0.8

        positions = np.array([
            [ half_length, -half_width],   # front-left
            [ half_length,  half_width],   # front-right
            [-half_length, -half_width],   # rear-left
            [-half_length,  half_width]    # rear-right
        ])

        T_row1, T_row2, T_row3 = [], [], []
        for (x, y) in positions:
            T_row1.extend([1, 0])
            T_row2.extend([0, 1])
            T_row3.extend([-y, x])
        self.T = np.vstack([T_row1, T_row2, T_row3])
        self.T_pseudo_inv = np.linalg.pinv(self.T)

        self.q_matrix = null_space(self.T)

    def allocate(self, tau, xi_p, dt):
        # xi_p = self.T_pseudo_inv @ tau
        xi_p_dot = (xi_p - self.last_xi_p) / dt
        self.last_xi_p = xi_p
        xi_0 = self.q_matrix @ self.theta
        xi_d = xi_p + xi_0
        xi_tilde = self.xi - xi_d
        v = -self.gamma * self.calculate_j_theta(xi_d).T
        theta_dot = v - self.mu * self.calculate_v_theta(xi_tilde).T
        self.theta += theta_dot * dt
        self.theta = np.clip(self.theta, self.theta_min, self.theta_max)
        kappa = self.calculate_kappa(xi_tilde, xi_p_dot, v)
        kappa = self.cbf(kappa, self.rho)
        self.xi += kappa * dt

        tau = self.T @ self.xi
        alpha = np.zeros(4)
        u = np.zeros(4)
        for i in range(self.num_thrusters):
            alpha[i] = np.arctan2(self.xi[2*i+1], self.xi[2*i])
            u[i] = np.linalg.norm(self.xi[2*i:2*i+2])

        return self.xi, u, alpha, tau
    
    def calculate_j_theta(self, xi_d: np.ndarray):
        j_theta = np.zeros_like(self.theta)
        a1 = 0.85
        a2 = 0.0083
        if self.power_management == False:
            for i in range(self.num_thrusters):
                xi_di = xi_d[2*i:2*i+2]
                xi_d_norm = np.linalg.norm(xi_di)

                if xi_d_norm != 0.0:
                    a_ref = self.reference_angles[i]
                    a_ref = np.array([np.cos(a_ref), np.sin(a_ref)])

                    j_theta += self.c[i] * (xi_di.T / xi_d_norm - self.lambda_ * a_ref.T) @ self.q_matrix[2*i:2*i+2, :]
        else:
            for i in range(self.num_thrusters):
                xi_di = xi_d[2*i:2*i+2]
                xi_d_norm = np.linalg.norm(xi_di)

                j_theta += (a1 + 2* a2 * xi_d_norm) * self.q_matrix[2*i:2*i+2, :].T @ xi_di / xi_d_norm
        return j_theta

    def calculate_v_theta(self, xi_tilde: np.ndarray):
        v_theta = np.zeros(5)
        for i in range(self.num_thrusters):
            v_theta -= self.c[i] * xi_tilde[2*i:2*i+2].T @ self.q_matrix[2*i:2*i+2, :]
        return v_theta
    
    def calculate_kappa(self, xi_tilde: np.ndarray, xi_p_dot: np.ndarray, v: np.ndarray):
        kappa = np.zeros(8)
        for i in range(self.num_thrusters):
            kappa[2*i:2*i+2] = -self.rate_limit * xi_tilde[2*i:2*i+2] / (np.linalg.norm(xi_tilde[2*i:2*i+2]) + self.zeta) + xi_p_dot[2*i:2*i+2] + self.q_matrix[2*i:2*i+2, :] @ v
            # kappa[2*i:2*i+2] = -self.rate_limit * xi_tilde[2*i:2*i+2] / 1.0 + 0.0 * xi_p_dot[2*i:2*i+2] + self.q_matrix[2*i:2*i+2, :] @ v
        return kappa
    
    def cbf(self, u, time_constant):
        kappa = np.zeros_like(u)
        for i in range(self.num_thrusters):
            u_min = -time_constant * (self.xi[2*i:2*i+2] - self.min_forces[2*i:2*i+2])
            u_max = time_constant * (self.max_forces[2*i:2*i+2] - self.xi[2*i:2*i+2])
            kappa[2*i:2*i+2] = np.clip(u[2*i:2*i+2], u_min, u_max)
        return kappa
