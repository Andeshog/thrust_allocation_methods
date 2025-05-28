import numpy as np
import numpy.typing as npt
import typing
import cvxpy as cp
from itertools import product
import scipy.linalg as la
import time
import enum


class ThrusterController:

    class Quadrant(enum.IntEnum):
        Q13 = 0
        Q24 = 1

    class ThrusterDynamicsQ13:

        def __init__(self, x: npt.ArrayLike = np.array([0.0, 0.0])):

            self.__x = x

            self.__q = 0

        def __b(self, x, q):
            """
            Hybrid barrier function:
            - If q=1, then B(x,1) = max(x1, x2)
            - If q=2, then B(x,2) = -min(x1, x2)

            - If q
            """
            x1, x2 = x
            if q == 1:
                return max(x1, x2)
            else:  # q == 2
                return -min(x1, x2)

        def __grad_b(self, x, q):
            """
            Gradient of the barrier function w.r.t. x, piecewise-defined.
            """
            x1, x2 = x
            if q == 1:
                # B(x)=max(x1,x2)
                if x1 > x2:
                    return np.array([1.0, 0.0])
                elif x2 > x1:
                    return np.array([0.0, 1.0])
                else:
                    # tie/boundary case
                    return np.array([0.5, 0.5])
            else:
                # B(x)=-min(x1,x2)
                if x1 < x2:
                    return np.array([-1.0, 0.0])
                elif x2 < x1:
                    return np.array([0.0, -1.0])
                else:
                    # tie/boundary case
                    return np.array([-0.5, -0.5])

        def __alpha(self, phi):
            """
            Simple class-K function alpha(phi) = gamma * phi.
            """
            gamma = 1.0
            return gamma * phi

        def __nominal_control(self, x, x_ref):
            """
            A nominal control law that tries to steer x toward x_ref.
            For example, a simple linear feedback: u_nom = - (x - x_ref).
            """
            return -(x - x_ref)

        def __safe_control(self, x, q, x_ref):
            """
            Enforce the CBF condition gradB(x,q) dot u <= -alpha(B(x,q)).
            1) Compute nominal control.
            2) If constraint is violated, subtract enough component along gradB
            to satisfy it.
            """
            u_nom = self.__nominal_control(x, x_ref)
            gb = self.__grad_b(x, q)
            b_val = self.__b(x, q)
            dot_val = gb.dot(u_nom)
            rhs = -self.__alpha(b_val)

            if dot_val <= rhs:
                # Already safe
                return u_nom
            else:
                # Need to adjust along gb
                correction = dot_val - rhs
                norm2 = gb.dot(gb)
                if abs(norm2) < 1e-10:
                    # fallback if gradient is near zero
                    return u_nom
                return u_nom - (correction / norm2) * gb

        def __jump_logic(self, x, q):
            """
            If B(x,q) >= 0, jump to the other discrete mode:
            1 -> 2, or 2 -> 1.
            """
            if self.__b(x, q) >= 0:
                return 2 if q == 1 else 1
            else:
                return q

        def control(self, x_ref: npt.ArrayLike, dt=0.01) -> npt.ArrayLike:

            # discrete jump check
            q_next = self.__jump_logic(self.__x, self.__q)
            if q_next != self.__q:
                self.__q = q_next

            # flow step
            u_s = self.__safe_control(self.__x, self.__q, x_ref)
            self.__x = self.__x + dt * u_s

            return self.__x

    class ThrusterDynamicsQ24:

        def __init__(self, x: npt.ArrayLike = np.array([0.0, 0.0])):

            self.__x = x

            self.__q = 0

        def __b(self, x, q):
            """
            Hybrid barrier function:
            - If q=1, then B(x,1) = max(x1, x2)
            - If q=2, then B(x,2) = -min(x1, x2)

            - If q
            """
            x1, x2 = x
            if q == 1:
                return max(x1, -x2)
            else:  # q == 2
                return max(-x1, x2)

        def __grad_b(self, x, q):
            """
            Gradient of the barrier function w.r.t. x, piecewise-defined.
            For mode 1: B(x,1)=max(x1, -x2)
            - If x1 > -x2, then grad = [1, 0]
            - If x1 < -x2, then grad = [0, -1]
            - Tie: average gradient [0.5, -0.5]
            For mode 2: B(x,2)=-min(x1, -x2)
            - If x1 < -x2, then grad = -[1, 0] = [-1, 0]
            - If x1 > -x2, then grad = -[0, -1] = [0, 1]
            - Tie: average gradient [-0.5, 0.5]
            """
            x1, x2 = x
            if q == 1:
                # For B(x,1)=max(x1,-x2)
                if x1 > -x2:
                    return np.array([1.0, 0.0])
                elif x1 < -x2:
                    return np.array([0.0, -1.0])
                else:
                    return np.array([0.5, -0.5])
            else:
                # For B(x,2) = -min(x1,-x2)
                if -x1 > x2:
                    return np.array([-1.0, 0.0])
                elif -x1 < x2:
                    return np.array([0.0, 1.0])
                else:
                    return np.array([-0.5, 0.5])

        def __alpha(self, phi):
            """
            Simple class-K function alpha(phi) = gamma * phi.
            """
            gamma = 1.0
            return gamma * phi

        def __nominal_control(self, x, x_ref):
            """
            A nominal control law that tries to steer x toward x_ref.
            For example, a simple linear feedback: u_nom = - (x - x_ref).
            """
            return -(x - x_ref)

        def __safe_control(self, x, q, x_ref):
            """
            Enforce the CBF condition gradB(x,q) dot u <= -alpha(B(x,q)).
            1) Compute nominal control.
            2) If constraint is violated, subtract enough component along gradB
            to satisfy it.
            """
            u_nom = self.__nominal_control(x, x_ref)
            gb = self.__grad_b(x, q)
            b_val = self.__b(x, q)
            dot_val = gb.dot(u_nom)
            rhs = -self.__alpha(b_val)

            if dot_val <= rhs:
                # Already safe
                return u_nom
            else:
                # Need to adjust along gb
                correction = dot_val - rhs
                norm2 = gb.dot(gb)
                if abs(norm2) < 1e-10:
                    # fallback if gradient is near zero
                    return u_nom
                return u_nom - (correction / norm2) * gb

        def __jump_logic(self, x, q):
            """
            If B(x,q) >= 0, jump to the other discrete mode:
            1 -> 2, or 2 -> 1.
            """
            if self.__b(x, q) >= 0:
                return 2 if q == 1 else 1
            else:
                return q

        def control(self, x_ref: npt.ArrayLike, dt=0.01) -> npt.ArrayLike:

            # discrete jump check
            q_next = self.__jump_logic(self.__x, self.__q)
            if q_next != self.__q:
                self.__q = q_next

            # flow step
            u_s = self.__safe_control(self.__x, self.__q, x_ref)
            self.__x = self.__x + dt * u_s

            return self.__x

    def __init__(
        self, x: npt.ArrayLike = np.array([0.0, 0.0]), mode: Quadrant = Quadrant.Q13
    ):

        if mode == self.Quadrant.Q13:
            self.__dynamics = self.ThrusterDynamicsQ13(x)
        else:
            self.__dynamics = self.ThrusterDynamicsQ24(x)

    def control(self, x_ref: npt.ArrayLike, dt=0.01) -> npt.ArrayLike:
        return self.__dynamics.control(x_ref, dt)


class PSQPAllocator:

    FORCE_LIMIT = 300.0  # Newtons
    NUM_THRUSTERS = 4
    MODE_CHANGE_THRESHOLD = 1e1  # Newtons
    HYSTERESIS_DURATION = 1  # seconds
    NEUTRAL_CONFIGURATION = 0

    def __init__(self):

        self.__last_mode_change = time.time()

        self.__current_configuration = self.NEUTRAL_CONFIGURATION

        self.__fore_port = ThrusterController(mode=ThrusterController.Quadrant.Q24)
        self.__fore_starboard = ThrusterController(mode=ThrusterController.Quadrant.Q13)
        self.__aft_port = ThrusterController(mode=ThrusterController.Quadrant.Q24)
        self.__aft_starboard = ThrusterController(mode=ThrusterController.Quadrant.Q13)

        self.__ts_last_command = time.time()

    def __solve_sqp(
        self, tau: npt.ArrayLike = np.array([0.0, 0.0, 0.0])
    ) -> typing.List[typing.Dict[str, typing.Any]]:
        """
        Solve the thrust allocation problem by enumerating each thruster's mode.
        Returns a list of solution dictionaries (one per combination).
        """
        L = 1.8
        W = 0.8
        num_thrusters = self.NUM_THRUSTERS
        thruster_positions = [
            (-L, W),  # Thruster 1: top-left
            (L, W),  # Thruster 2: top-right
            (L, -W),  # Thruster 3: bottom-right
            (-L, -W),  # Thruster 4: bottom-left
        ]

        Fx_d, Fy_d, Mz_d = tau

        # 2. Desired Forces/Moment as Parameters
        Fx_param = cp.Parameter(value=Fx_d)
        Fy_param = cp.Parameter(value=Fy_d)
        Mz_param = cp.Parameter(value=Mz_d)

        # 3. Thruster Limits
        T_max = self.FORCE_LIMIT  # nominal maximum thrust magnitude

        # 4. Decision Variables
        Tix = cp.Variable(num_thrusters)
        Tiy = cp.Variable(num_thrusters)
        eFx = cp.Variable(nonneg=True)
        eFy = cp.Variable(nonneg=True)
        eMz = cp.Variable(nonneg=True)
        # delta = cp.Variable(num_thrusters, nonneg=True)

        # 5. Sign Parameters for Quadrant Constraints
        sign_x = [cp.Parameter() for _ in range(num_thrusters)]
        sign_y = [cp.Parameter() for _ in range(num_thrusters)]

        # Define mode-specific sign pairs for each thruster.
        # For example, for thruster 0, mode 0 gives (-1,+1) and mode 1 gives (+1,-1)
        thruster_modes = [
            [(+1, -1), (-1, +1)],  # thruster 0
            [(-1, -1), (+1, +1)],  # thruster 1
            [(-1, +1), (+1, -1)],  # thruster 2
            [(+1, +1), (-1, -1)],  # thruster 3
        ]

        # 6. Common Constraints: Enforce that each thrust component has the prescribed sign.
        constraints = []
        for i in range(num_thrusters):
            constraints += [sign_x[i] * Tix[i] >= 0, sign_y[i] * Tiy[i] >= 0]

        # Force/moment matching with slack
        constraints += [
            (cp.sum(Tix) - Fx_param) <= eFx,
            -(cp.sum(Tix) - Fx_param) <= eFx,
            (cp.sum(Tiy) - Fy_param) <= eFy,
            -(cp.sum(Tiy) - Fy_param) <= eFy,
        ]

        moment_expr = 0
        for i in range(num_thrusters):
            x_i, y_i = thruster_positions[i]
            moment_expr += x_i * Tiy[i] - y_i * Tix[i]
        constraints += [
            (moment_expr - Mz_param) <= eMz,
            -(moment_expr - Mz_param) <= eMz,
        ]

        # Thruster magnitude constraints (with slack)
        for i in range(num_thrusters):
            constraints += [
                Tix[i] <= T_max,  # + delta[i],
                -Tix[i] <= T_max,  # + delta[i],
                Tiy[i] <= T_max,  # + delta[i],
                -Tiy[i] <= T_max,  # + delta[i],
            ]

        # 7. Objective: minimize energy plus heavy penalties on slack.
        W_slack = 1e4
        objective = cp.Minimize(cp.sum(Tix**2 + Tiy**2) + W_slack * (eFx + eFy + eMz))

        problem = cp.Problem(objective, constraints)

        combinations = list(product([0, 1], repeat=num_thrusters))

        # 8. Solve for Each Combination of Modes
        solutions = []
        for idx, combo in enumerate(combinations):
            # Set the sign parameters based on the mode combination.
            for i in range(num_thrusters):
                sx, sy = thruster_modes[i][combo[i]]
                sign_x[i].value = sx
                sign_y[i].value = sy

            try:
                problem.solve(solver=cp.OSQP, verbose=False)
                sol_data = {
                    "combo_index": idx,
                    "status": problem.status,
                    "objective": problem.value,
                    "Tix": Tix.value.tolist(),
                    "Tiy": Tiy.value.tolist(),
                    "eFx": eFx.value,
                    "eFy": eFy.value,
                    "eMz": eMz.value,
                }
            except Exception as e:
                sol_data = {
                    "combo_index": idx,
                    "status": problem.status,
                    "objective": problem.value,
                    "Tix": Tix.value.tolist(),
                    "Tiy": Tiy.value.tolist(),
                    "eFx": eFx.value,
                    "eFy": eFy.value,
                    "eMz": eMz.value,
                }

            sol_data["ts"] = time.time()

            solutions.append(sol_data)

        return solutions

    def __choose_solution(self, solutions: typing.List[dict]) -> dict:

        should_change = False
        should_change_to = self.NEUTRAL_CONFIGURATION

        sorted_solutions = sorted(
            solutions, key=lambda x: x["objective"], reverse=False
        )

        # Check the norm of the slack variables
        pref_slack_norm = la.norm(
            [
                solutions[self.NEUTRAL_CONFIGURATION]["eFx"],
                solutions[self.NEUTRAL_CONFIGURATION]["eFy"],
                solutions[self.NEUTRAL_CONFIGURATION]["eMz"],
            ]
        )

        current_slack_norm = la.norm(
            [
                solutions[self.__current_configuration]["eFx"],
                solutions[self.__current_configuration]["eFy"],
                solutions[self.__current_configuration]["eMz"],
            ]
        )

        optimum_slack_norm = la.norm(
            [
                sorted_solutions[0]["eFx"],
                sorted_solutions[0]["eFy"],
                sorted_solutions[0]["eMz"],
            ]
        )

        # print(
        #     f"current: {current_slack_norm:2e}, threshold: {self.MODE_CHANGE_THRESHOLD:2e}, pref: {pref_slack_norm:2e}, opt: {optimum_slack_norm:2e}"
        # )
        if time.time() - self.__last_mode_change > self.HYSTERESIS_DURATION:

            if pref_slack_norm < current_slack_norm:
                should_change = True
                should_change_to = self.NEUTRAL_CONFIGURATION
            elif current_slack_norm < self.MODE_CHANGE_THRESHOLD:
                should_change = False
            else:
                if optimum_slack_norm < current_slack_norm:
                    should_change = True
                    should_change_to = sorted_solutions[0]["combo_index"]
                else:
                    should_change = False
        else:
            should_change = False

        if should_change:
            print(
                f"Changing mode from {self.__current_configuration} to {should_change_to}"
            )
            self.__last_mode_change = time.time()
            self.__current_configuration = should_change_to

        return solutions[self.__current_configuration]

    def __control_thrusters(self, solution: dict) -> npt.ArrayLike:

        Tix = np.array(solution["Tix"])
        Tiy = np.array(solution["Tiy"])

        cmdix, cmdiy = Tix, Tiy

        dt = np.abs(self.__ts_last_command - solution["ts"])
        self.__ts_last_command = solution["ts"]

        cmdix[0], cmdiy[0] = self.__fore_port.control([Tix[0], Tiy[0]], dt=dt)
        cmdix[1], cmdiy[1] = self.__fore_starboard.control([Tix[1], Tiy[1]], dt=dt)
        cmdix[2], cmdiy[2] = self.__aft_port.control([Tix[2], Tiy[2]], dt=dt)
        cmdix[3], cmdiy[3] = self.__aft_starboard.control([Tix[3], Tiy[3]], dt=dt)

        solution["cmdix"] = cmdix
        solution["cmdiy"] = cmdiy

        return solution

    def solve(self, tau: npt.ArrayLike, ts_override: float = None) -> dict:

        solutions = self.__solve_sqp(tau=tau)

        # Override the timestamp for testing purposes
        if ts_override is not None:
            for sol in solutions:
                sol["ts"] = ts_override

        best_solution = self.__choose_solution(solutions)

        control_command = self.__control_thrusters(best_solution)

        return control_command