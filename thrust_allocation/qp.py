import numpy as np
import cvxpy as cp

class QPAllocator:
    def __init__(self,
                 half_length=1.8,
                 half_width=0.8,
                 u_bound=300.0,
                 max_rate=1.0,
                 max_force_rate=50.0,
                 W=None,
                 Q=None,
                 beta=1.1):
        """
        Initializes the QP-based thrust allocator.

        Parameters:
            half_length (float): Half the vessel length (m).
            half_width (float): Half the vessel width (m).
            u_bound (float): Thruster force bound (N); forces are in [-u_bound, u_bound].
            max_rate (float): Maximum allowed change in thruster angle (rad per timestep).
            max_force_rate (float): Maximum allowed change in thruster force (N per timestep).
            W (np.ndarray): Weight matrix for the force components (default: identity).
            Q (np.ndarray): Weight matrix for slack variables (default: 1000*I).
            beta (float): Weight for the extra scalar variable f̄.
        """
        self.half_length = half_length
        self.half_width = half_width
        # Construct thruster positions (4 thrusters)
        self.positions = np.array([
            [ half_length, -half_width],   # front-left
            [ half_length,  half_width],   # front-right
            [-half_length, -half_width],   # rear-left
            [-half_length,  half_width]    # rear-right
        ])
        self.n_thrusters = 4

        # Desired thruster angles (inward pointing as in the NLP model)
        self.alpha_des = np.array([3*np.pi/4, -3*np.pi/4, np.pi/4, -np.pi/4])
        # Initialize previous angles and forces (for rate limiting)
        self.alpha_prev = self.alpha_des.copy()
        self.u_prev = np.zeros(self.n_thrusters)

        # Static thruster force bounds (for the scalar force u)
        self.u_bound = u_bound
        self.u_lb = np.zeros(self.n_thrusters)
        self.u_ub = u_bound * np.ones(self.n_thrusters)

        # Static allowed azimuth sectors for each thruster (from NLP)
        # front-left: [π/2, π], front-right: [-π, -π/2],
        # rear-left: [0, π/2], rear-right: [-π/2, 0]
        self.static_alpha_lb = np.array([np.pi/2, -np.pi, 0, -np.pi/2])
        self.static_alpha_ub = np.array([np.pi, -np.pi/2, np.pi/2, 0])

        self.max_rate = max_rate
        self.max_force_rate = max_force_rate

        # QP cost function parameters.
        # Decision variable: z = [f; s; f_bar]
        #   f: extended force vector, with 2 components per thruster (Fx and Fy)
        #   s: slack variable (3-dimensional, one per DOF)
        #   f_bar: scalar bounding the magnitude of each f component.
        self.W = W if W is not None else np.eye(2*self.n_thrusters)
        self.Q = Q if Q is not None else 1000 * np.eye(3)
        self.beta = beta

        # Build the constant part of the QP model.
        # Extended configuration matrix T: maps f to net force/moment.
        T_row1, T_row2, T_row3 = [], [], []
        for (x, y) in self.positions:
            T_row1.extend([1, 0])
            T_row2.extend([0, 1])
            T_row3.extend([-y, x])
        self.T = np.vstack([T_row1, T_row2, T_row3])

        print(f"T_matrix: {self.T}")
        
        # Dimension of decision variable z: 
        #   f: 2*n_thrusters, s: 3, f_bar: 1  => total = 2*n + 3 + 1.
        self.z_dim = 2*self.n_thrusters + 3 + 1
        
        # Precompute the constant block in the objective.
        H_f = 2 * self.W  # (2*n_thrusters x 2*n_thrusters)
        H_s = 2 * self.Q  # (3 x 3)
        H_fb = 1e-6 * np.eye(1)  # small regularization for f_bar

        self.H_const = np.block([
            [H_f,                           np.zeros((2*self.n_thrusters, 3)),   np.zeros((2*self.n_thrusters, 1))],
            [np.zeros((3, 2*self.n_thrusters)), H_s,                         np.zeros((3, 1))],
            [np.zeros((1, 2*self.n_thrusters + 3)), H_fb]
        ])

        self.c_const = np.zeros((self.z_dim,))
        self.c_const[-1] = self.beta

    def allocate(self, tau_desired, dt):
        """
        Solves the QP thrust allocation problem for a given desired net force/moment.

        Parameters:
            tau_desired (array_like): Desired net force/moment (3-element vector).
            dt (float): Time step (for rate limiting).

        Returns:
            f_ext (np.ndarray): (n_thrusters x 2) extended force vector ([Fx, Fy] per thruster).
            thrusts (np.ndarray): (n_thrusters,) thrust magnitudes.
            angles (np.ndarray): (n_thrusters,) computed azimuth angles (rad).
        """
        n = self.n_thrusters
        delta_u_max = self.max_force_rate * dt

        # Compute dynamic allowed angle bounds per thruster (intersection of static bounds and rate limits).
        dynamic_alpha_lb = self.static_alpha_lb
        dynamic_alpha_ub = self.static_alpha_ub
        
        # Compute dynamic force (magnitude) bounds for each thruster.
        dynamic_u_lb = np.maximum(self.u_lb, self.u_prev - delta_u_max)
        dynamic_u_ub = np.minimum(self.u_ub, self.u_prev + delta_u_max)
        
        # --- Build the QP model dynamically ---
        # Decision variable: z = [f; s; f_bar]
        z = cp.Variable(self.z_dim)
        f = z[:2*n]             # extended force vector (length 2*n)
        s = z[2*n:2*n+3]         # slack variables (3, one per DOF)
        f_bar = z[-1]           # scalar variable bounding force components

        # Equality constraint: T*f - s == tau_desired.
        eq_constr = [self.T @ f - s == tau_desired]

        constr = []
        # 1. Force component bounds: for each thruster, for both Fx and Fy.
        for i in range(n):
            idx_fx = 2*i
            idx_fy = 2*i + 1
            constr.append(f[idx_fx] <= dynamic_u_ub[i])
            constr.append(f[idx_fx] >= -dynamic_u_ub[i])
            constr.append(f[idx_fy] <= dynamic_u_ub[i])
            constr.append(f[idx_fy] >= -dynamic_u_ub[i])
        
        # 2. f_bar bounds: |f[j]| <= f_bar for all components.
        for i in range(2*n):
            constr.append(f[i] - f_bar <= 0.0)
            constr.append(-f[i] - f_bar <= 0.0)
        constr.append(f_bar >= 1.0)
        
        # 3. Allowed azimuth sector constraints.
        # For each thruster, enforce that the resulting angle (from f) lies between the dynamic bounds.
        # This is done via two linear inequalities:
        #   np.cos(th_min)*Fy - np.sin(th_min)*Fx >= 0
        #   Fx*np.sin(th_max) - Fy*np.cos(th_max) >= 0
        epsilon = 1e-4  # small tolerance
        for i in range(n):
            Fx = f[2*i]
            Fy = f[2*i + 1]
            th_min = dynamic_alpha_lb[i]
            th_max = dynamic_alpha_ub[i]
            constr.append(np.cos(th_min) * Fy - np.sin(th_min) * Fx >= -epsilon)
            constr.append(Fx * np.sin(th_max) - np.cos(th_max) * Fy >= -epsilon)

        
        # Combine constraints.
        constraints = eq_constr + constr

        # Define the objective: minimize 0.5*zᵀH*z + cᵀz.
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
                # Force the thruster's magnitude to min_force and set its angle to the desired inward angle.
                angles[i] = self.alpha_des[i]
                f_ext[i, :] = np.array([np.cos(self.alpha_des[i]), np.sin(self.alpha_des[i])]) * min_force
                thrusts[i] = min_force
        
        self.alpha_prev = angles
        self.u_prev = thrusts

        tau_actual = self.T @ f_ext.reshape(8, 1)
        tau_actual = tau_actual.flatten()

        f_ext = f_ext.flatten()
        
        return f_ext, thrusts, angles, tau_actual

