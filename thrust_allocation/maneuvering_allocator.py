import numpy as np
from scipy.linalg import null_space
from dataclasses import dataclass

@dataclass
class ManeuveringParams:
    dt: float = 0.1
    gamma: float = 0.1
    mu: float = 10.0
    rho: float = 0.1
    zeta: float = 0.0001
    lambda_: float = 1.0
    rate_limit: float = 100.0
    theta_min: float = -50.0
    theta_max: float = 50.0
    max_force: float = 300.0
    power_management: bool = False

class ManeuveringAllocator:
    def __init__(self, params: ManeuveringParams):
        self.power_management = params.power_management

        self.gamma = params.gamma
        self.mu = params.mu
        self.w_matrix = np.eye(8)
        self.last_xi_p = np.zeros(8)
        self.xi = np.zeros(8)
        self.theta = np.zeros(5)
        self.rho = params.rho
        self.zeta = params.zeta
        self.rate_limit = params.rate_limit
        self.c = np.ones(4)
        self.lambda_ = params.lambda_

        self.theta_min = params.theta_min
        self.theta_max = params.theta_max
        self.dt = params.dt

        max_forces = params.max_force
        self.min_forces = np.array([-max_forces, 1.0, -max_forces, -max_forces, 1.0, 1.0, 1.0, -max_forces])
        self.max_forces = np.array([-1.0, max_forces, -1.0, -1.0, max_forces, max_forces, max_forces, -1.0])
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

    def allocate(self, tau: np.ndarray, xi_p: np.ndarray):
        # xi_p = self.T_pseudo_inv @ tau
        xi_p_dot = (xi_p - self.last_xi_p) / self.dt
        self.last_xi_p = xi_p
        xi_0 = self.q_matrix @ self.theta
        xi_d = xi_p + xi_0
        xi_tilde = self.xi - xi_d
        v = -self.gamma * self.calculate_j_theta(xi_d).T
        theta_dot = v - self.mu * self.calculate_v_theta(xi_tilde).T
        self.theta += theta_dot * self.dt
        self.theta = np.clip(self.theta, self.theta_min, self.theta_max)
        kappa = self.calculate_kappa(xi_tilde, xi_p_dot, v)
        kappa = self.cbf(kappa, self.rho)
        self.xi += kappa * self.dt

        tau = self.T @ self.xi
        alpha = np.zeros(4)
        u = np.zeros(4)
        for i in range(self.num_thrusters):
            alpha[i] = np.arctan2(self.xi[2*i+1], self.xi[2*i])
            u[i] = np.linalg.norm(self.xi[2*i:2*i+2])

        return self.xi, u, alpha, tau
    
    def calculate_j_theta(self, xi_d: np.ndarray):
        j_theta = np.zeros_like(self.theta)
        a1_pos = 0.85
        a2_pos = 0.0083
        a1_neg = -1.45
        a2_neg = 0.0115
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

                if i == 2 or i == 3:  # Rear thrusters
                    if xi_di[0] > 0: # Positive thrust
                        a1 = a1_pos
                        a2 = a2_pos
                        sign = 1
                    else:  # Negative thrust
                        a1 = a1_neg
                        a2 = a2_neg
                        sign = -1
                else:  # Front thrusters
                    if xi_di[0] > 0: # Negative thrust
                        a1 = a1_neg
                        a2 = a2_neg
                        sign = -1
                    else:  # Positive thrust
                        a1 = a1_pos
                        a2 = a2_pos
                        sign = 1

                j_theta += (sign * a1 + 2* a2 * xi_d_norm) * self.q_matrix[2*i:2*i+2, :].T @ xi_di / (xi_d_norm + 0.001)
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
        return kappa
    
    def cbf(self, u, time_constant):
        kappa = np.zeros_like(u)
        for i in range(self.num_thrusters):
            u_min = -time_constant * (self.xi[2*i:2*i+2] - self.min_forces[2*i:2*i+2])
            u_max = time_constant * (self.max_forces[2*i:2*i+2] - self.xi[2*i:2*i+2])
            kappa[2*i:2*i+2] = np.clip(u[2*i:2*i+2], u_min, u_max)
        return kappa
