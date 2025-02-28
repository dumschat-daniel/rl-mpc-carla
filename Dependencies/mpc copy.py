import casadi as ca
import numpy as np
import math
import matplotlib.pyplot as plt

class MPC:
    def __init__(self):
        self.dt = 0.1
        self.N = 2
        self.L = 2.875
        self.target_speed = 25 / 3.6
        self.v_max = 100 / 3.6
        self.v_min = 0
        self.a_max = 3
        self.a_min = -8
        self.delta_max = math.radians(70)
        self.delta_min = -self.delta_max
        self.e_y_max = 3
        self.e_y_min = -self.e_y_max
        self.e_theta_max = math.radians(100)
        self.e_theta_min = -self.e_theta_max
        self.opti = ca.Opti()

        self.X = None
        self.U = None
        self.P = None
        self.obj = 0

    
        self.param = {}  
        self.opts = {}

        
        self.constraints = []

        self.spline_x, self.spline_y = None, None
        
        self.warm_start = True 
        self.previous_solution_X = None 
        self.previous_solution_U = None  

        self.predicted_trajectories = []    

        self.n_steps = 0
        self.t_total = 0
        self.iterations_total = 0

    def setup_MPC(self):
        self.init_system_model()
        self.init_constraints()
        self.compute_optimization_cost()
        self.init_solver()


    def init_system_model(self):
        
        s = ca.MX.sym('s')         
        e_y = ca.MX.sym('e_y')      
        e_theta = ca.MX.sym('e_psi') 
        v = ca.MX.sym('v')          

        delta = ca.MX.sym('delta') 
        a = ca.MX.sym('a')          

        state = ca.vertcat(s, e_y, e_theta, v)
        control = ca.vertcat(delta, a)

        self.n_states = state.size1()
        self.n_controls = control.size1()

        self.k = ca.MX.sym('k')

        s_dot = v * ca.cos(e_theta) / (1 - self.k * e_y)
        e_y_dot = v * ca.sin(e_theta)
        e_theta_dot = (v / self.L) * ca.tan(delta) - self.k * s_dot
        v_dot = a

        rhs = ca.vertcat(s_dot, e_y_dot, e_theta_dot, v_dot)
        #discretized_rhs = state + self.dt * rhs
        self.f = ca.Function('f', [state, control, self.k], [rhs])

        self.X = self.opti.variable(self.n_states, self.N + 1)  # States over horizon
        self.U = self.opti.variable(self.n_controls, self.N)   # Controls over horizon
        self.P = self.opti.parameter(self.n_states + self.n_states * self.N + self.N)  # Parameters

        self.obj = 0  


    def set_reference_trajectory(self, spline_x, spline_y):
        self.spline_x = spline_x
        self.spline_y = spline_y


    def init_constraints(self):
        # System dynamics constraints
        for i in range(self.N):
            k_i = self.P[self.n_states + self.n_states * self.N + i]
            x_next = self.X[:, i] + self.dt * self.f(self.X[:, i], self.U[:, i], k_i)
            self.opti.subject_to(self.X[:, i + 1] == x_next)

        # State and control limits
        self.opti.subject_to(self.opti.bounded(self.delta_min, self.U[0, :], self.delta_max))  # Steering limits
        self.opti.subject_to(self.opti.bounded(self.a_min, self.U[1, :], self.a_max))         # Acceleration limits
        self.opti.subject_to(self.opti.bounded(self.e_y_min, self.X[1, :], self.e_y_max))     # Lateral error bounds
        self.opti.subject_to(self.opti.bounded(self.e_theta_min, self.X[2, :], self.e_theta_max)) # Heading error bounds
        self.opti.subject_to(self.opti.bounded(self.v_min, self.X[3, :], self.v_max))         # Speed bounds


    def init_solver(self):
        #self.opti.solver('ipopt', {}, self.opts)
        self.opti.solver('ipopt', {
        'ipopt.tol': 1e-8,  # Overall convergence tolerance
        'ipopt.acceptable_tol': 1e-6,  # Acceptable tolerance for early stopping
        'ipopt.acceptable_iter': 10,  # Allow a few iterations at acceptable tolerance
        'ipopt.print_level': 0,  # Print level for solver output
        'print_time': False,  # Enable or disable timing
    })



    def compute_optimization_cost(self):
        Q = np.diag([50, 100, 100, 25])  # weights for [s, e_y, e_theta, v]
        R = np.diag([10, 10])          # weights for [delta, a]

        for i in range(self.N):
            state_ref = self.P[self.n_states + i * self.n_states : self.n_states + (i + 1) * self.n_states]
            state_error = self.X[:, i] - state_ref
            self.obj += ca.mtimes([state_error.T, Q, state_error])

            control = self.U[:, i]
            self.obj += ca.mtimes([control.T, R, control])


        self.opti.minimize(self.obj)


    def init_mpc_start_conditions(self, current_state, P):
        
        self.opti.subject_to(self.X[:, 0] == current_state)
        #print("Current state before MPC:", current_state)
        #print("Reference trajectory start (P[:4]):", P[:4])
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

    def generate_reference_trajectory(self, current_state, spline_x, spline_y):

        # get reference speed
        current_speed = current_state[3]
        reference_speeds = np.minimum(
            self.target_speed,
            current_speed + np.arange(self.N) * self.a_max
        )

        s_values = [current_state[0]]  # Start with the current progress along the polynomial
        for i in range(self.N):
            #s_next = s_values[-1] + reference_speeds[i] * self.dt
            s_next = s_values[-1] + reference_speeds[i]
            s_values.append(s_next)
        #s_values = np.array(s_values[:-1])  # Convert to NumPy array
        s_values = np.array(s_values[1:])

        # Get derivatives from the spline
        dx_ref = spline_x(s_values, 1)  # First derivative of x
        dy_ref = spline_y(s_values, 1)  # First derivative of y
        ddx_ref = spline_x(s_values, 2)  # Second derivative of x
        ddy_ref = spline_y(s_values, 2)  # Second derivative of y

        # Calculate curvature (kappa)
        curvature = (dx_ref * ddy_ref - dy_ref * ddx_ref) / (dx_ref**2 + dy_ref**2)**1.5
        curvature = np.clip(curvature, -1e3, 1e3)  # Clip to avoid extreme values


        # Generate reference states
        reference_states = np.column_stack((
            s_values,              # s
            np.zeros(self.N),       # e_y (on the polynomial)
            np.zeros(self.N),       # e_theta (aligned with the polynomial)
            reference_speeds          # v (capped reference speed)
        ))

        # Overwrite the first state with the current state
        reference_states[0] = current_state

        #print(reference_states)
        return reference_states, curvature

    def compare_trajectory_with_splines(self, generated_s_values, generated_x_ref, generated_y_ref, spline_x, spline_y, distances):
        """
        Compares generated reference trajectory points (x_ref, y_ref) with the spline values.
        Args:
            generated_s_values (np.ndarray): Generated s values from the reference trajectory.
            generated_x_ref (np.ndarray): x_ref values from the generated trajectory.
            generated_y_ref (np.ndarray): y_ref values from the generated trajectory.
            spline_x (CubicSpline): Cubic spline for x.
            spline_y (CubicSpline): Cubic spline for y.
            distances (np.ndarray): Distances array for the spline.
        """
        # Uniformly distributed s_values for the spline
        s_min = generated_s_values[0]
        s_max = generated_s_values[-1]
        uniform_s_values = np.linspace(s_min, s_max, len(generated_s_values))  # Match number of points for comparison


        # Clip generated_s_values to ensure they are within the spline range
        generated_s_values_clipped = np.clip(generated_s_values, distances[0], distances[-1])

        # Compute x, y for the spline
        spline_x_ref_uniform = spline_x(uniform_s_values)
        spline_y_ref_uniform = spline_y(uniform_s_values)

        # Compute x, y for the generated points on the spline
        generated_spline_x_ref = spline_x(generated_s_values_clipped)
        generated_spline_y_ref = spline_y(generated_s_values_clipped)

        # Debugging differences
        print(f"s_values (Generated): {generated_s_values}")
        print(f"s_values (Uniform Spline): {uniform_s_values}")
        print(f"x_diff: {np.abs(generated_x_ref - generated_spline_x_ref)}")
        print(f"y_diff: {np.abs(generated_y_ref - generated_spline_y_ref)}")

        # Visualization
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.plot(spline_x_ref_uniform, spline_y_ref_uniform, label='Spline Polynomial (Uniform)', marker='o', linestyle='-')
        plt.plot(generated_x_ref, generated_y_ref, label='Generated Trajectory', marker='x', linestyle='--')
        #plt.plot(generated_spline_x_ref, generated_spline_y_ref, label='Generated Points on Spline', marker='s', linestyle=':')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Comparison of Generated Trajectory and Polynomial Spline')
        plt.grid()
        plt.show()
    
    def construct_parameters(self, current_state, reference_trajectory, curvature):
        if len(reference_trajectory) != self.N or len(curvature) != self.N:
            raise ValueError("Reference trajectory and curvature lengths must match N.")
        
        ref_flat = reference_trajectory.flatten() 
        return ca.DM(np.hstack((current_state, ref_flat, curvature)))

    def solve(self, current_state):

        reference_trajectory, curvature = self.generate_reference_trajectory(
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

            #self.predicted_trajectories.append(optimal_X)
            self.predicted_trajectories.append(optimal_X[:, 0])
            #stats = self.opti.stats()

            self.n_steps += 1
            #self.t_total += stats['t_wall_total']
            #self.iterations_total += stats['iter_count']
            #print("CONTROL INPUT", optimal_U[:, 0])
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


    def visualize_accumulated_trajectories(self, gnss_sensor):
        """
        Visualizes the accumulated predicted trajectories from the MPC over multiple steps.
        Args:
            gnss_sensor (GNSS_sensor): GNSS sensor object containing the reference trajectory splines.
        """
        # Flatten accumulated predicted trajectories
        #all_predicted_s = np.concatenate([traj[0, :] for traj in self.predicted_trajectories])
        #all_predicted_e_y = np.concatenate([traj[1, :] for traj in self.predicted_trajectories])
        all_predicted_s = np.array([state[0] for state in self.predicted_trajectories])  # Predicted `s`
        all_predicted_e_y = np.array([state[1] for state in self.predicted_trajectories])
        predicted_x = self.spline_x(all_predicted_s) - all_predicted_e_y * np.sin(self.spline_x(all_predicted_s, 1))
        predicted_y = self.spline_y(all_predicted_s) + all_predicted_e_y * np.cos(self.spline_y(all_predicted_s, 1))

        # Generate reference trajectory
        reference_s = np.linspace(0, gnss_sensor.distances[-1], len(gnss_sensor.distances))
        reference_x = gnss_sensor.spline_x(reference_s)
        reference_y = gnss_sensor.spline_y(reference_s)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(reference_x, reference_y, label="Reference Trajectory", linestyle="--", color="gray")
        plt.plot(predicted_x, predicted_y, label="Predicted Trajectories", linestyle="-", marker="o", color="blue")
        plt.scatter(predicted_x[0], predicted_y[0], label="Start Point", color="green", zorder=5)
        plt.scatter(predicted_x[-1], predicted_y[-1], label="End Point", color="red", zorder=5)

        # Labels and Legends
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("Accumulated MPC Predicted Trajectories vs. Reference Trajectory")
        plt.legend()
        plt.grid()
        plt.axis("equal")
        plt.show()


