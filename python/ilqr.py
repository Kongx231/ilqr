import numpy as np
# import sympy as sp

class ilqr:
    def __init__(self,init_state,target_state,initial_guess,dt,start_time,end_time,f_disc,A,B,Q_k,R_k,Q_T,parameters,n_iterations):
        self.init_state_ = init_state
        self.target_state_ = target_state
        self.inputs_ = initial_guess
        self.n_states_ = np.shape(init_state)[0]
        self.n_inputs_ = np.shape(initial_guess)[1]

        self.dt_ = dt
        self.start_time_ = start_time
        self.end_time_ = end_time
        self.time_span_ = np.arange(start_time, end_time, dt).flatten()
        self.n_timesteps_ = np.shape(self.time_span_)[0]
        # Dynamics
        self.f_ = f_disc
        self.A_ = A
        self.B_ = B
        # Weighting
        self.Q_k_ = Q_k
        self.R_k_ = R_k
        self.Q_T_ = Q_T
        self.parameters_ = parameters

        # Max iterations
        self.n_iterations_ = n_iterations

    def rollout(self):
        states = np.zeros((self.n_timesteps_ + 1, self.n_states_))
        inputs = np.zeros((self.n_timesteps_, self.n_inputs_))
        current_state = self.init_state_

        for ii in range(0,self.n_timesteps_):
            current_input = self.inputs_[ii,:]
            next_state = self.f_(current_state, current_input, self.dt_, self.parameters_).flatten()
            # Store states and inputs
            states[ii + 1,:] = next_state
            inputs[ii,:] = current_input # in case we have a control law, we store the input used
            # Update the current state
            current_state = next_state

        # Store the trajectory(states, inputs)
        self.states_ = states
        self.inputs_ = inputs
        return states, inputs

    def compute_cost(self,states,inputs):
        # Initialize cost
        total_cost = 0.0
        for ii in range(0,self.n_timesteps_):
            current_x = states[ii,:] # Not being used currently
            current_u = inputs[ii,:].flatten()

            current_cost = current_u.T@self.R_k_@current_u # Right now only considering cost in input
            total_cost = total_cost+current_cost
        # Compute terminal cost
        terminal_difference = (self.target_state_-states[-1,:]).flatten()
        terminal_cost = terminal_difference.T@self.Q_T_@terminal_difference
        total_cost = total_cost+terminal_cost
        return total_cost

    def backwards_pass(self):
        # First compute initial conditions (end boundary condition)
        # Value function hessian and gradient
        V_xx = self.Q_T_
        end_difference = (self.states_[-1, :] - self.target_state_).flatten()
        end_difference = end_difference.flatten()  # Make sure its the right dimension
        V_x = self.Q_T_@end_difference

        # Initialize storage variables
        k_trj = np.zeros((self.n_timesteps_,self.n_inputs_))
        K_trj = np.zeros((self.n_timesteps_,self.n_inputs_,self.n_states_))

        # Initialize cost reduction
        expected_cost_reduction = 0
        expected_cost_reduction_grad = 0
        expected_cost_reduction_hess = 0

        # for loop backwards in time
        for idx in reversed(range(0, self.n_timesteps_)):
            # Grab the current variables in the trajectory
            current_x = self.states_[idx,:]
            current_u = self.inputs_[idx,:]

            # R_k_updated
            # Define the expansion coefficients and the loss gradients
            l_xx = self.Q_k_ # For now zeros, can add in a target to track later on
            l_uu = self.R_k_

            l_x = self.Q_k_@np.zeros(self.n_states_).flatten() # For now zeros, can add in a target to track later on
            l_u = self.R_k_@(current_u).flatten()

            # Get the jacobian of the discretized dynamics
            A_k = self.A_(current_x, current_u, self.dt_, self.parameters_)
            B_k = self.B_(current_x, current_u, self.dt_, self.parameters_)

            Q_x = l_x + A_k.T@V_x
            Q_u = l_u+B_k.T@V_x
            Q_ux = B_k.T@V_xx@A_k
            Q_uu = l_uu + B_k.T@V_xx@B_k
            Q_xx = l_xx+A_k.T@V_xx@A_k

            # Compute gains
            Q_uu_inv = np.linalg.inv(Q_uu) # This can sometimes go singular
            k = -Q_uu_inv@Q_u
            K = -Q_uu_inv@Q_ux

            # Store gains
            k_trj[idx,:] = k
            K_trj[idx,:,:] = K

            # Update the expected reduction
            current_cost_reduction_grad = -Q_u.T@k
            current_cost_reduction_hess = 0.5 * k.T @ (Q_uu) @ (k)
            current_cost_reduction = current_cost_reduction_grad + current_cost_reduction_hess

            expected_cost_reduction_grad +=  current_cost_reduction_grad
            expected_cost_reduction_hess +=  current_cost_reduction_hess
            expected_cost_reduction += + current_cost_reduction

            # Update hessian and gradient for value function (If we arent using regularization we can simplify this computation)
            V_x = Q_x +K.T@Q_uu@k +K.T@Q_u + Q_ux.T@k
            V_xx = (Q_xx+Q_ux.T@K+K.T@Q_ux+K.T@Q_uu@K)

        # Store expected cost reductions
        self.expected_cost_reduction_grad_ = expected_cost_reduction_grad
        self.expected_cost_reduction_hess_ = expected_cost_reduction_hess
        self.expected_cost_reduction_ = expected_cost_reduction

        # Store gain schedule
        self.k_feedforward_ = k_trj
        self.K_feedback_ = K_trj
        return (k_trj,K_trj,expected_cost_reduction)

    def forwards_pass(self, learning_rate):
        states = np.zeros((self.n_timesteps_ + 1, self.n_states_))
        inputs = np.zeros((self.n_timesteps_, self.n_inputs_))
        current_state = self.init_state_

        # set the first state to be  the initial
        states[1,:] = current_state
        for ii in range(0,self.n_timesteps_):
            # Get the current gains and compute the feedforward and feedback terms
            current_feedforward = learning_rate * self.k_feedforward_[ii,:]
            current_feedback = self.K_feedback_[ii,:,:]@(current_state - self.states_[ii,:])
            current_input = self.inputs_[ii,:] + current_feedback + current_feedforward

            # simulate forward
            next_state = self.f_(current_state, current_input, self.dt_, self.parameters_).flatten()
            # Store states and inputs
            states[ii + 1,:] = next_state
            inputs[ii,:] = current_input.flatten()

            # Update the current state
            current_state = next_state
        return (states,inputs)
    def solve(self):
        # Compute the rollout to get the initial trajectory with the
        # initial guess
        [states,inputs] = self.rollout()
        # Compute the current cost of the initial trajectory
        current_cost = self.compute_cost(states,inputs)
        
        learning_speed = 0.95 # This can be modified, 0.95 is very slow
        low_learning_rate = 0.05 # if learning rate drops to this value stop the optimization
        low_expected_reduction = 1e-3 # Determines optimality
        armijo_threshold = 0.1 # Determines if current line search solve is good (this is typically labeled as "c")
        for ii in range(0,self.n_iterations_):
            print('Starting iteration: ',ii)
            # Compute the backwards pass
            (k_feedforward,K_feedback,expected_reduction) = self.backwards_pass()
            
            if(abs(expected_reduction)<low_expected_reduction):
                # If the expected reduction is low, then end the
                # optimization
                print("Stopping optimization, optimal trajectory")
                break
            learning_rate = 1
            armijo_flag = 0
            # Execute linesearch until the armijo condition is met (for
            # now just check if the cost decreased) TODO add real
            # armijo condition
            while(learning_rate > 0.05 and armijo_flag == 0):
                # Compute forward pass
                (new_states,new_inputs)=self.forwards_pass(learning_rate)
                new_cost = self.compute_cost(new_states,new_inputs)

                # Calculate armijo condition
                cost_difference = (current_cost - new_cost)
                expected_cost_redu = learning_rate*self.expected_cost_reduction_grad_ + learning_rate*learning_rate*self.expected_cost_reduction_hess_
                armijo_flag = cost_difference/expected_cost_redu > armijo_threshold
                if(armijo_flag == 1):
                    # Accept the new trajectory if armijo condition is
                    # met
                    current_cost = new_cost
                    self.states_ = new_states
                    self.inputs_ = new_inputs
                else:
                    # If no improvement, decrease the learning rate
                    learning_rate = learning_speed*learning_rate
                    print('Reducing learning rate to: ',learning_rate)
            if(learning_rate<low_learning_rate):
                # If learning rate is low, then stop optimization
                print("Stopping optimization, low learning rate")
                break
        # Return the current trajectory
        states = self.states_
        inputs = self.inputs_
        return states,inputs,k_feedforward,K_feedback,current_cost



        
