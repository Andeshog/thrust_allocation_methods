import casadi as ca
import numpy as np

class NLPAllocator:
    def __init__(self,
                 half_length=1.8,
                 half_width=0.8,
                 u_bound=400.0,
                 w_angle=1000.0,
                 w_neg=10.0,
                 max_rate=0.5,
                 max_force_rate=200.0,
                 w_alpha_change=50000.0):
        """
        Initializes the thrust allocator.

        Parameters:
            half_length (float): Half the length of the vessel (m).
            half_width (float): Half the width of the vessel (m).
            u_bound (float): Bound on thruster force (N); forces are in [-u_bound, u_bound].
            w_angle (float): Weight for penalty term that keeps thruster angles near a "desired" angle.
            w_neg (float): Weight for negative force penalty.
            max_rate (float): Maximum allowable change in thruster angle per time step (rad).
            max_force_rate (float): Maximum allowable change in thruster force per time step (N).
            w_alpha_change (float): Additional penalty weight on changes in alpha vs. alpha_prev.
        """
        # Vessel geometry and thruster positions
        self.half_length = half_length
        self.half_width = half_width
        self.positions = np.array([
            [ half_length,  half_width],   # front-left
            [ half_length, -half_width],   # front-right
            [-half_length,  half_width],   # rear-left
            [-half_length, -half_width]    # rear-right
        ])
        self.n_thrusters = 4

        # Desired thruster angles (inward facing)
        self.alpha_des = np.array([
            3*np.pi/4,
            -3*np.pi/4,
             np.pi/4,
            -np.pi/4
        ])
        # Keep track of previous solution for warm starting
        self.alpha_prev = self.alpha_des.copy()
        self.u_prev = np.zeros(self.n_thrusters)

        # Force bounds
        self.u_lb = -u_bound * np.ones(self.n_thrusters)
        self.u_ub =  u_bound * np.ones(self.n_thrusters)

        # Angle bounds
        self.static_alpha_lb = np.array([ np.pi/2, -np.pi,     0,      -np.pi/2])
        self.static_alpha_ub = np.array([     np.pi, -np.pi/2, np.pi/2,        0])

        # Max allowable angle/force rates
        self.max_rate = max_rate
        self.max_force_rate = max_force_rate

        # Weights in the cost function
        self.w_angle = w_angle
        self.w_neg = w_neg
        self.w_alpha_change = w_alpha_change

        # Build the NLP
        self._build_nlp()

        # Initial guess
        self.x0 = np.concatenate([self.alpha_des, np.zeros(self.n_thrusters)])

    def _build_nlp(self):
        """Builds the CasADi NLP for thrust allocation."""

        n = self.n_thrusters

        # Symbolic decision variables
        alpha = ca.SX.sym('alpha', n)
        u     = ca.SX.sym('u',     n)

        # We'll treat tau (3 elements) + alpha_prev (n elements) as the parameters
        #   p = [ tau_sym (3), alpha_prev_sym (n) ]
        tau_sym        = ca.SX.sym('tau', 3)
        alpha_prev_sym = ca.SX.sym('alpha_prev', n)

        # Net force/moment
        Fx = 0
        Fy = 0
        Mz = 0
        for i in range(n):
            Fx += ca.cos(alpha[i]) * u[i]
            Fy += ca.sin(alpha[i]) * u[i]
            Mz += (self.positions[i, 0] * ca.sin(alpha[i])
                 - self.positions[i, 1] * ca.cos(alpha[i])) * u[i]
        F_net = ca.vertcat(Fx, Fy, Mz)

        # Cost function
        #  1) 0.5 * sum(u^2) 
        #  2) w_angle   * sum( (alpha - alpha_des)^2 )
        #  3) w_neg     * sum( fmax(0, -u)^2 )
        #  4) w_alpha_change * sum( (alpha - alpha_prev)^2 )
        cost = 0.5 * ca.sumsqr(u)
        cost += self.w_angle * ca.sumsqr(alpha - ca.DM(self.alpha_des))
        cost += self.w_neg   * ca.sumsqr(ca.fmax(0, -u))
        # Here is the new penalty on changes from alpha_prev
        cost += self.w_alpha_change * ca.sumsqr(alpha - alpha_prev_sym)

        # Equality constraints: we want F_net == tau_desired
        g = F_net - tau_sym

        # Decision variable vector
        x = ca.vertcat(alpha, u)

        # Put everything into the NLP dict
        nlp = {
            'x': x,
            'f': cost,
            'g': g,
            'p': ca.vertcat(tau_sym, alpha_prev_sym)
        }

        opts = {"print_time": False, "ipopt.print_level": 0}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    def allocate(self, tau_desired, dt):
        """
        Solves the thrust allocation problem for a given desired net force/moment.

        Parameters:
            tau_desired (array_like): Desired net force/moment (3-element vector).
            dt (float): Time step for rate limiting.

        Returns:
            alpha_opt (np.ndarray): Optimal thruster angles.
            u_opt     (np.ndarray): Optimal thruster forces.
            tau_out   (np.ndarray): Achieved net force/moment.
        """
        n = self.n_thrusters

        # Rate limits (dynamic bounds)
        delta_alpha_max = self.max_rate * dt
        delta_u_max     = self.max_force_rate * dt

        # Dynamic alpha bounds
        alpha_lb_dynamic = np.maximum(self.static_alpha_lb,
                                      self.alpha_prev - delta_alpha_max)
        alpha_ub_dynamic = np.minimum(self.static_alpha_ub,
                                      self.alpha_prev + delta_alpha_max)

        # Dynamic force bounds
        u_lb_dynamic = np.maximum(self.u_lb, self.u_prev - delta_u_max)
        u_ub_dynamic = np.minimum(self.u_ub, self.u_prev + delta_u_max)

        # Full decision variable bounds
        lb_x = np.concatenate([alpha_lb_dynamic, u_lb_dynamic])
        ub_x = np.concatenate([alpha_ub_dynamic, u_ub_dynamic])

        # Equality constraints g == 0
        lb_g = np.zeros(3)
        ub_g = np.zeros(3)

        # The parameter vector includes tau_desired and alpha_prev
        p = np.concatenate([tau_desired, self.alpha_prev])

        # Solve the NLP
        sol = self.solver(
            x0=self.x0,
            lbx=lb_x,
            ubx=ub_x,
            lbg=lb_g,
            ubg=ub_g,
            p=p
        )
        x_opt = sol['x'].full().flatten()

        # Extract alpha and u from the solution
        alpha_opt = x_opt[:n]
        u_opt     = x_opt[n:]

        # Compute the achieved net force/moment
        Fx = np.sum(np.cos(alpha_opt) * u_opt)
        Fy = np.sum(np.sin(alpha_opt) * u_opt)
        Mz = np.sum((self.positions[:, 0] * np.sin(alpha_opt)
                   - self.positions[:, 1] * np.cos(alpha_opt)) * u_opt)
        tau_out = np.array([Fx, Fy, Mz])

        # Update warm start + "previous" values
        self.x0        = x_opt
        self.alpha_prev = alpha_opt
        self.u_prev     = u_opt

        return alpha_opt, u_opt, tau_out
