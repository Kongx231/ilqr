classdef ilqr < handle
    % This class sets up the optimization problem with the constructor and
    % solve will return the optimal set of inputs and gains to achieve a desired target state.
    % Note that the discrete dynamics of the system and it's Jacobians are
    % required to be functions ready to be called with
    % inputs(state,input,dt,parameters)
    properties
        init_state_
        states_
        target_state_
        inputs_
        dt_
        start_time_
        end_time_
        time_span_
        A_
        B_
        Q_k_
        R_k_
        Q_T_
        parameters_
        K_feedback_
        k_feedforward_
        n_states_
        n_inputs_
        f_
        n_timesteps_
        n_iterations_
    end
    methods
        % Constructor
        function self = ilqr(init_state,target_state,initial_guess,dt,start_time,end_time,f_disc,A,B,Q_k,R_k,Q_T,parameters,n_iterations)
            self.init_state_ = init_state;
            self.target_state_ = target_state;
            self.inputs_ = initial_guess;
            self.n_states_ = size(init_state,1);
            self.n_inputs_ = size(initial_guess,2);
            
            self.dt_ = dt;
            self.start_time_ = start_time;
            self.end_time_ = end_time;
            self.time_span_ = start_time:dt:end_time;
            self.n_timesteps_ = size(self.time_span_,2);
            % Dynamics
            self.f_ = f_disc;
            self.A_ = A;
            self.B_ = B;
            % Weighting
            self.Q_k_ = Q_k;
            self.R_k_ = R_k;
            self.Q_T_ = Q_T;
            self.parameters_ = parameters;
            
            % Max iterations
            self.n_iterations_ = n_iterations;
        end
        function [states,inputs,k_feedforward,K_feedback,current_cost] = solve(self)
            % Compute the rollout to get the initial trajectory with the
            % initial guess
            [states,inputs] = self.rollout();
            % Compute the current cost of the initial trajectory
            current_cost = self.compute_cost(states,inputs);
            
            learning_speed = 0.95; % This can be modified, 0.95 is very slow
            low_learning_rate = 0.05; % if learning rate drops to this value stop the optimization
            low_expected_reduction = 1e-3; % Determines optimality
            for ii = 1:self.n_iterations_
                disp(['Starting iteration: ',num2str(ii)]);
                % Compute the backwards pass
                [k_feedforward,K_feedback,expected_reduction] = self.backwards_pass();
                
                if(abs(expected_reduction)<low_expected_reduction)
                    % If the expected reduction is low, then end the
                    % optimization
                    disp("Stopping optimization, optimal trajectory");
                    break;
                end
                learning_rate = 1;
                armijo_flag = 0;
                % Execute linesearch until the armijo condition is met (for
                % now just check if the cost decreased) TODO add real
                % armijo condition
                while(learning_rate > 0.05 && armijo_flag == 0)
                    % Compute forward pass
                    [new_states,new_inputs]=forwards_pass(self,learning_rate);
                    new_cost = self.compute_cost(new_states,new_inputs);
                    armijo_flag = current_cost - new_cost > 0; % Not real armijo condition, just checking if cost decreased
                    if(armijo_flag == 1)
                        % Accept the new trajectory if armijo condition is
                        % met
                        current_cost = new_cost;
                        self.states_ = new_states;
                        self.inputs_ = new_inputs;
                    else
                        % If no improvement, decrease the learning rate
                        learning_rate = learning_speed*learning_rate;
                        disp(['Reducing learning rate to: ',num2str(learning_rate)]);
                    end
                end
                if(learning_rate<low_learning_rate)
                    % If learning rate is low, then stop optimization
                    disp("Stopping optimization, low learning rate");
                    break;
                end
            end
            % Return the current trajectory
            states = self.states_;
            inputs = self.inputs_;
        end
        function total_cost = compute_cost(self,states,inputs)
            % Initialize cost
            total_cost = 0.0;
            for ii = 1:self.n_timesteps_
                current_x = states(ii,:)'; % Not being used currently
                current_u = inputs(ii,:)';
                
                current_cost = current_u'*self.R_k_*current_u; % Right now only considering cost in input
                total_cost = total_cost+current_cost;
            end
            % Compute terminal cost
            terminal_cost = (self.target_state_-states(end,:)')'*self.Q_T_*(self.target_state_-states(end,:)');
            total_cost = total_cost+terminal_cost;
        end
        function [states,inputs] = rollout(self)
            states = zeros(self.n_timesteps_+1,self.n_states_);
            inputs = zeros(self.n_timesteps_,self.n_inputs_);
            current_state = self.init_state_;
            
            for ii=1:self.n_timesteps_
                current_input = self.inputs_(ii,:)';
                next_state = self.f_(current_state,current_input,self.dt_,self.parameters_);
                % Store states and inputs
                states(ii+1,:) = next_state';
                inputs(ii,:) = current_input'; % in case we have a control law, we store the input used
                % Update the current state
                current_state = next_state;
            end
            % Store the trajectory (states,inputs)
            self.states_ = states;
            self.inputs_= inputs;
        end
        function [K_trj,k_trj,expected_cost_redu] = backwards_pass(self)
            % Initialize feedforward gains
            k_trj = zeros(size(self.inputs_));
            K_trj = zeros(size(self.inputs_,1),size(self.inputs_,2),size(self.states_,2));
            % Initialize expected cost reduction
            expected_cost_redu = 0;
            
            % Iitialize gradient and hessian of the value function
            V_x = self.Q_T_*(self.states_(end,:)'-self.target_state_);
            V_xx = self.Q_T_;
            
            for ii = flip(1:self.n_timesteps_) % Go backwards in time
                % Get the current state and input
                current_x = self.states_(ii,:)';
                current_u = self.inputs_(ii,:)';
                
                % Get the gradient and hessian of the current cost
                l_x = zeros(size(current_x)); % Defined as zero right now because there is no desired trajectory
                l_xx = self.Q_k_; % Q_k should also be zero here
                l_u = self.R_k_*current_u;
                l_uu = self.R_k_;
                
                % Get the jacobian of the discretized dynamics
                A_k = self.A_(current_x,current_u,self.dt_,self.parameters_);
                B_k =self.B_(current_x,current_u,self.dt_,self.parameters_);
                
                % Compute the coefficient expansion terms
                Q_x = l_x+A_k'*V_x;
                Q_u = l_u+B_k'*V_x;
                Q_xx = l_xx+A_k'*V_xx*A_k;
                Q_ux = B_k'*(V_xx)*A_k;
                Q_uu = l_uu+B_k'*(V_xx)*B_k;
                
                % Compute the gains
                k = -(Q_uu)\Q_u;
                K = -(Q_uu)\Q_ux;
                
                % Update the gradient and hessian of the value function
                V_x = Q_x +K'*Q_uu*k +K'*Q_u + Q_ux'*k;
                V_xx = (Q_xx+Q_ux'*K+K'*Q_ux+K'*Q_uu*K);
                
                % Store the gains
                k_trj(ii,:) = k';
                K_trj(ii,:,:) = K;
                
                current_cost_reduction = -Q_u'*k - 0.5 * k'*(Q_uu)*k;
                expected_cost_redu = expected_cost_redu+current_cost_reduction;
            end
            self.k_feedforward_= k_trj;
            self.K_feedback_ = K_trj;
        end
        function [states,inputs]=forwards_pass(self,learning_rate)
            states = zeros(self.n_timesteps_+1,self.n_states_);
            inputs = zeros(self.n_timesteps_,self.n_inputs_);
            current_state = self.init_state_;
            
            for ii=1:self.n_timesteps_
                % Get the current gains and compute the feedforward and
                % feedback terms
                current_feedforward = learning_rate*self.k_feedforward_(ii,:)';
                current_feedback = reshape(self.K_feedback_(ii,:,:),self.n_inputs_,self.n_states_)*(current_state-self.states_(ii,:)');
                current_input = self.inputs_(ii,:)' + current_feedback + current_feedforward;
                
                % simualte forward
                next_state = self.f_(current_state,current_input,self.dt_,self.parameters_);
                % Store states and inputs
                states(ii+1,:) = next_state';
                inputs(ii,:) = current_input';
                % Update the current state
                current_state = next_state;
            end
        end
    end
end

