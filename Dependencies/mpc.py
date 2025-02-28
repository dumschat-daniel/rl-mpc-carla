import casadi as ca
import numpy as np
import math

class MPC:
    """MPC Implementation. Using Bicycle Model with Ackermann Steering to compute high level optimal actions."""
    def __init__(self):
        # MPC Parameters
        self.dt = 0.1
        self.N = 2
        self.L = 2.875
        self.target_speed = 25 / 3.6
        self.v_max = 180 / 3.6
        self.v_min = 0
        self.a_max = 3
        self.a_min = -8
        self.delta_max = math.radians(70)
        self.delta_min = -self.delta_max
        self.e_y_max = 4
        self.e_y_min = -self.e_y_max
        self.e_psi_max = math.radians(55)
        self.e_psi_min = -self.e_psi_max
        self.opti = ca.Opti()

        self.X = None
        self.U = None
        self.P = None
        self.obj = 0

        self.spline_x, self.spline_y = None, None
        
        self.warm_start = True 
        self.previous_solution_X = None 
        self.previous_solution_U = None  

        self.predicted_trajectories = []    


    def setup_MPC(self):
        """setup the MPC Problem."""
        self.init_system_model()
        self.init_constraints()
        self.compute_optimization_cost()
        self.init_solver()


    def init_system_model(self):
        """Initializes the System Dynamics and Bicycle model based on s, e_y, e_psi, v."""

        s = ca.MX.sym('s')         
        e_y = ca.MX.sym('e_y')      
        e_psi = ca.MX.sym('e_psi') 
        v = ca.MX.sym('v')          

        delta = ca.MX.sym('delta') 
        a = ca.MX.sym('a')          

        state = ca.vertcat(s, e_y, e_psi, v)
        control = ca.vertcat(delta, a)

        self.n_states = state.size1()
        self.n_controls = control.size1()

        self.k = ca.MX.sym('k')

        s_dot = v * ca.cos(e_psi) / (1 - self.k * e_y)
        e_y_dot = v * ca.sin(e_psi)
        e_psi_dot = (v / self.L) * ca.tan(delta) - self.k * s_dot
        v_dot = a

        rhs = ca.vertcat(s_dot, e_y_dot, e_psi_dot, v_dot)
        self.f = ca.Function('f', [state, control, self.k], [rhs])

        self.X = self.opti.variable(self.n_states, self.N + 1)  # States over horizon
        self.U = self.opti.variable(self.n_controls, self.N)   # Controls over horizon
        self.P = self.opti.parameter(self.n_states + self.n_states * self.N + self.N)  # Parameters

        self.obj = 0  


    def set_reference_trajectory(self, spline_x, spline_y):
        """sets the Reference Trajectory from gnss sensor. Later used to set optimal States over the Horizon."""
        self.spline_x = spline_x
        self.spline_y = spline_y


    def init_constraints(self):
        """initializes System dynamics constraints."""
        #Euler-Diskretisierung
        for i in range(self.N):
            k_i = self.P[self.n_states + self.n_states * self.N + i]
            x_next = self.X[:, i] + self.dt * self.f(self.X[:, i], self.U[:, i], k_i)
            self.opti.subject_to(self.X[:, i + 1] == x_next)

        # State and control limits
        self.opti.subject_to(self.opti.bounded(self.delta_min, self.U[0, :], self.delta_max))  # Steering limits
        self.opti.subject_to(self.opti.bounded(self.a_min, self.U[1, :], self.a_max))         # Acceleration limits
        self.opti.subject_to(self.opti.bounded(self.e_y_min, self.X[1, :], self.e_y_max))     # Lateral error bounds
        self.opti.subject_to(self.opti.bounded(self.e_psi_min, self.X[2, :], self.e_psi_max)) # Heading error bounds
        self.opti.subject_to(self.opti.bounded(self.v_min, self.X[3, :], self.v_max))         # Speed bounds


    def init_solver(self):
        """initializes the Solver and the problem tolerance."""
        self.opti.solver('ipopt', {
        'ipopt.tol': 1e-12,  # Overall convergence tolerance
        'ipopt.acceptable_tol': 1e-10,  # Acceptable tolerance for early stopping
        'ipopt.acceptable_iter': 10,  # Allow a few iterations at acceptable tolerance
        'ipopt.print_level': 0,  # Print level for solver output
        'print_time': False,  # Enable or disable timing
    })



    def compute_optimization_cost(self):
        """Sets up the optimization cosst based on the Weight formulation """
        Q = np.diag([50, 500, 100, 25])  # weights for [s, e_y, e_psi, v]
        R = np.diag([10, 10])          # weights for [delta, a]

        for i in range(self.N):
            state_ref = self.P[self.n_states + i * self.n_states : self.n_states + (i + 1) * self.n_states]
            state_error = self.X[:, i] - state_ref
            self.obj += ca.mtimes([state_error.T, Q, state_error])

            control = self.U[:, i]
            self.obj += ca.mtimes([control.T, R, control])


        self.opti.minimize(self.obj)


    def init_mpc_start_conditions(self, current_state, P):
        """Defines start condition (first state==current state) and uses warm start"""
        self.opti.subject_to(self.X[:, 0] == current_state)
        if self.warm_start and self.previous_solution_X is not None and self.previous_solution_U is not None:
            # Warm start with the previous solution
            self.opti.set_initial(self.X[:, :-1], self.previous_solution_X[:, 1:])  # Shift states
            self.opti.set_initial(self.X[:, -1], self.previous_solution_X[:, -1])  # Keep last state
            self.opti.set_initial(self.U[:, :-1], self.previous_solution_U[:, 1:])  # Shift controls
            self.opti.set_initial(self.U[:, -1], self.previous_solution_U[:, -1])  # Keep last control
        else:
            for i in range(1, self.N + 1):
                state_ref = P[self.n_states + (i - 1) * self.n_states : self.n_states + i * self.n_states]
                self.opti.set_initial(self.X[:, i], state_ref)
            self.opti.set_initial(self.U, 0) 

    def generate_reference_positions_along_trajectory(self, current_state, spline_x, spline_y):
        """Use Trajectory to compute optimal States along the Trajectory."""
        # get reference speed
        current_speed = current_state[3]
        reference_speeds = np.minimum(
            self.target_speed,
            current_speed + np.arange(self.N) * self.a_max
        )

        s_values = [current_state[0]]  # Start with the current progress along the polynomial
        for i in range(self.N):
            s_next = s_values[-1] + reference_speeds[i]
            s_values.append(s_next)
        s_values = np.array(s_values[1:])

        # Get derivatives from the splined
        dx_ref = spline_x(s_values, 1) 
        dy_ref = spline_y(s_values, 1)  
        ddx_ref = spline_x(s_values, 2)  
        ddy_ref = spline_y(s_values, 2) 

        # Calculate curvature (kappa)
        curvature = (dx_ref * ddy_ref - dy_ref * ddx_ref) / (dx_ref**2 + dy_ref**2)**1.5
        curvature = np.clip(curvature, -1e3, 1e3)  # Clip to avoid extreme values


        # Generate reference states
        reference_states = np.column_stack((
            s_values,              # s
            np.zeros(self.N),       # e_y (on the polynomial)
            np.zeros(self.N),       # e_psi (aligned with the polynomial)
            reference_speeds          # v (capped reference speed based on theoretical possible speed)
        ))

        # Overwrite the first state with the current state
        reference_states[0] = current_state

        return reference_states, curvature


    def construct_parameters(self, current_state, reference_trajectory, curvature):
        if len(reference_trajectory) != self.N or len(curvature) != self.N:
            raise ValueError("Reference trajectory and curvature lengths must match N.")
        
        ref_flat = reference_trajectory.flatten() 
        return ca.DM(np.hstack((current_state, ref_flat, curvature)))

    def solve(self, current_state):
        """generates the reference Positions and constructs the Problem. Solves the Problem and returns the solution"""
        reference_trajectory, curvature = self.generate_reference_positions_along_trajectory(
        current_state=current_state,
        spline_x=self.spline_x,
        spline_y=self.spline_y
    )
        P = self.construct_parameters(current_state, reference_trajectory, curvature)
        self.opti.set_value(self.P, P)

        self.init_mpc_start_conditions(current_state, P)

        try:
            solution = self.opti.solve()

            # Extract optimal control inputs and trajectory
            optimal_U = solution.value(self.U)
            optimal_X = solution.value(self.X)

            self.previous_solution_X = optimal_X
            self.previous_solution_U = optimal_U

            self.predicted_trajectories.append(optimal_X[:, 0])
            
            return optimal_U[:, 0], optimal_X
        except RuntimeError as e:
            print(f"MPC failed to solve: {e}")
            # Debugging variables
            '''
            print("Debugging variables at the time of failure:")
            print("X:", self.opti.debug.value(self.X))
            print("U:", self.opti.debug.value(self.U))
            print("P:", self.opti.debug.value(self.P))
            print("G:", self.opti.debug.value(self.opti.g))  # Constraints values
            print("F:", self.opti.debug.value(self.opti.f))  # Objective value
            '''
            return None, None


