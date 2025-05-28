import numpy as np
from math import atan2, sqrt, pi, tanh
from casadi import SX, vertcat, nlpsol, cos, sin

class ThrustAllocation:
    def __init__(self):
        self.init_params()

    def init_params(self) -> None:
        self.l_x: float = None
        self.l_y: float = None
        self.motor_rpm_factor: float = None
        self.p_propeller_rpm_to_command : float = None
        self.p_thrust_to_propeller_rpm : np.ndarray = None
        self.single_thruster_min_force: float = None
        self.single_thruster_max_force: float = None

        half_length=1.8
        half_width=0.8

        self.positions = np.array([
            [ half_length, -half_width],   # front-left
            [ half_length,  half_width],   # front-right
            [-half_length, -half_width],   # rear-left
            [-half_length,  half_width]    # rear-right
        ])

        T_row1, T_row2, T_row3 = [], [], []
        for (x, y) in self.positions:
            T_row1.extend([1, 0])
            T_row2.extend([0, 1])
            T_row3.extend([-y, x])
        self.T = np.vstack([T_row1, T_row2, T_row3])

    def allocate_thrust(self, tau: np.ndarray, last_thrust: np.ndarray, last_angles: np.ndarray) -> np.ndarray:
        f = SX.sym('f', 4)
        a = SX.sym('a', 4)

        tau_casadi = [tau[0], tau[1], -tau[2]]

        # lx = [self.l_x, self.l_x, -self.l_x, -self.l_x]
        # ly = [-self.l_y, self.l_y, self.l_y, -self.l_y]
        # lx = [self.l_x] * 4
        # ly = [self.l_y] * 4
        lx = [self.l_y] * 4
        ly = [self.l_x] * 4

        f_0 = last_thrust.flatten()
        a_0 = [3*pi/4, -3*pi/4, -pi/4, pi/4]

        f_obj = self.cost_function(f, f_0, a, a_0, lx, ly, tau_casadi)

        P = dict(f=f_obj, x=vertcat(f, a))
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        F = nlpsol('F', 'ipopt', P, opts)
        r = F(x0 = [f_0[0], f_0[1], f_0[2], f_0[3], a_0[0], a_0[1], a_0[2], a_0[3]],
		     lbx = [0, 0, 0, 0, 3 * pi/4 - pi/36, -3 * pi/4 - pi/36, -pi/4 - pi/36, pi/4 - pi/36],
		     ubx = [1000, 1000, 1000, 1000, 3 * pi/4 + pi/36, -3 * pi / 4 + pi/36, -pi/4 + pi/36, pi/4 + pi/36])
		
        forces_angles = np.array(r['x'])

        forces_angles[0:4] * self.motor_rpm_factor

        forces_extended = np.zeros(8)
        for i in range(4):
            forces_extended[i * 2] = forces_angles[i] * cos(forces_angles[i + 4])
            forces_extended[i * 2 + 1] = forces_angles[i] * sin(forces_angles[i + 4])
        tau = self.T @ forces_extended
        return forces_angles.flatten(), tau

    def cost_function(self, f, f_0, a, a_0, lx, ly, tau):
        term1 = (-f[0] * cos(pi - a[0]) - 
                  f[1] * cos(a[1] - pi) + 
                  f[2] * cos(2*pi - a[2]) + 
                  f[3] * cos(a[3]) - tau[0])**2
        
        term2 = (f[0] * sin(pi - a[0]) - 
                 f[1] * sin(a[1] - pi) - 
                 f[2] * sin(2 * pi - a[2]) + 
                 f[3] * sin(a[3]) - tau[1])**2
        
        term3 = 10 * (f[0] * sin(pi - a[0]) * lx[0] - f[0] * cos(pi - a[0]) * ly[0] - 
                      f[1] * sin(a[1] - pi) * lx[1] + f[1] * cos(a[1] - pi) * ly[1] + 
                      f[2] * sin(2 * pi - a[2]) * lx[2] - f[2] * cos(2 * pi - a[2]) * ly[2] - 
                      f[3] * sin(a[3]) * lx[3] + f[3] * cos(a[3]) * ly[3] - tau[2])**2
        
        term4 = 0.1 * ((f[0] - f_0[0])**2 + 
                       (f[1] - f_0[1])**2 + 
                       (f[2] - f_0[2])**2 + 
                       (f[3]-f_0[3])**2)
        
        term5 = 100 * ((a[0] - a_0[0])**2 + 
                       (a[1] - a_0[1])**2 + 
                       (a[2] - a_0[2])**2 + 
                       (a[3] - a_0[3])**2)
		    
        f_obj = term1 + term2 + term3 + term4 + term5

        return f_obj
    
    def thrust_to_rpm(self, F):
        coeff_fit_pos = np.array([-0.005392628724401, 4.355160257309755, 0])
        coeff_fit_neg = np.array([0.009497954053629, 5.891321999522722, 0])
        tmax = 430
        tmin = -330
		
        if F < 0:
            rpm = np.polyval(coeff_fit_neg, F)
        else:
            rpm = np.polyval(coeff_fit_pos, F)

        if F > tmax:
            rpm = np.polyval(coeff_fit_pos, tmax)

        if F < tmin:
            rpm = np.polyval(coeff_fit_neg, tmin)

        return rpm
    
    def rpm_to_command(self, rpm: float) -> float:
        p = self.p_propeller_rpm_to_command

        command = np.clip(p * rpm, -1000, 1000)

        return command
    
    def thrust_to_command(self, thrust: float) -> float:
        rpm = self.thrust_to_rpm(thrust)
        command = self.rpm_to_command(rpm)
        return command