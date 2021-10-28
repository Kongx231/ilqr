classdef ilqr_mpc < handle
    % This class sets up the optimization problem with the constructor and
    % solve_ilqr will return the optimal set of inputs and gains to track a
    % target trajectory
    % Note that the discrete dynamics of the system and it's Jacobians are
    % required to be functions ready to be called with
    % inputs(state,input,dt,parameters)
    properties
        current_state_
        states_
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
        target_states_
        target_inputs_
        horizon_
        current_idx_
        % Expected cost reductions
        expected_cost_redu_
        expected_cost_redu_grad_
        expected_cost_redu_hess_
    end
    methods
        % Constructor
        % Need to add in tracking
        function self = ilqr_mpc(target_states,initial_guess,dt,horizon,f_disc,A,B,Q_k,R_k,Q_T,parameters,n_iterations)
            self.target_states_ = target_states;
            self.target_inputs_ = initial_guess;
            self.n_states_ = size(target_states,2);
            self.n_inputs_ = size(initial_guess,2);
            
            self.dt_ = dt;
            
            self.n_timesteps_ = horizon;
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
        function [states,inputs,k_feedforward,K_feedback,current_cost] = solve_ilqr(self,current_idx,current_state)
            
            
            % If first time solving, then compute the backwards pass for
            % gains
            if(current_idx == 1)
                % Store the current state and current idx
                self.current_state_ = current_state;
                self.current_idx_ = current_idx;
                % Set the reference trajectory as the actual
                self.inputs_ = self.target_inputs_(current_idx:self.n_timesteps_,:);
                self.states_ = self.target_states_(current_idx:(self.n_timesteps_+1),:);
                % First compute backwards pass along the reference
                % trajectory to get the gain schedule
                [k_feedforward,K_feedback,expected_reduction] = ...
                    self.backwards_pass(self.states_,self.inputs_);
            else
                % TODO PAD THIS CORRECTLY WHEN NOT SOLVING EVERY TIME
                % Calculate last time was solve_ilqr was called
                span_from_last_call = current_idx-self.current_idx_;
                % Store the current state and current idx
                self.current_state_ = current_state;
                self.current_idx_ = current_idx;
                
                
                % Probably just use self.current_idx_-current_idx to get it
                
                
                % Update the warm start to include the end of the
                % trajectory
                
                % use the reference to pad
                
                if(span_from_last_call>self.n_timesteps_)
                    % Check if last call is larger than horizon (THIS SHOULD
                % NEVER BE THE CASE)
                end
                horizon_end_idx = current_idx+self.n_timesteps_;
                
%                 self.inputs_ = [self.inputs_((span_from_last_call+1):end,:);
%                     self.target_inputs_((horizon_end_idx-span_from_last_call):(horizon_end_idx-1),:)];
%                 self.states_ = [self.states_((span_from_last_call+1):end,:);
%                     self.target_states_((horizon_end_idx-span_from_last_call+1):horizon_end_idx,:)];

                self.inputs_ = [self.inputs_((span_from_last_call+1):end,:);
                    self.target_inputs_((horizon_end_idx-span_from_last_call):(horizon_end_idx-1),:)];
                self.states_ = [self.states_((span_from_last_call+1):end,:);
                    self.target_states_((horizon_end_idx-span_from_last_call+1):horizon_end_idx,:)];
                
                %                 % repeat the last index instead of using the reference to
                %                 pad (worse convergence)
                %                 self.inputs_ = [self.inputs_(2:end,:); self.inputs_(end,:)];
                %                 self.states_ = [self.states_(2:end,:); self.states_(end,:)];
                
                % Repeat the gain (or we can have the gain solved for the
                % entire trajectory and use that)
%                 self.K_feedback_ = [self.K_feedback_(2:end,:,:); self.K_feedback_(end,:,:)];
                repeated_gain = repmat(self.K_feedback_(end,:,:),span_from_last_call,1,1);
                self.K_feedback_ = [self.K_feedback_((span_from_last_call):end,:,:);repeated_gain];
            end
            
            % Compute the rollout to get the initial trajectory with the
            % initial guess (use forward pass with 0 learning rate to use
            % the feedback gains)
            [new_states,new_inputs]=self.forwards_pass(0); % Learning rate 0 to use no feed forward terms (this might be wrong)
            %             animate_pendulum(new_states)
            
            % Compute the current cost of the initial trajectory
            % Store the current trajectory and cost
            current_cost = self.compute_cost(new_states,new_inputs);
            self.states_ = new_states;
            self.inputs_ = new_inputs;
            
            learning_speed = 0.95; % This can be modified, 0.95 is very slow
            low_learning_rate = 0.05; % if learning rate drops to this value stop the optimization
            
            low_expected_reduction = 1e-3; % Determines optimality
            armijo_threshold = 0.1; % Determines if current line search solve is good (this is typically labeled as "c")
            
            for ii = 1:self.n_iterations_
                disp(['Starting iteration: ',num2str(ii)]);
                % Compute the backwards pass
                [k_feedforward,K_feedback,expected_reduction] = self.backwards_pass(new_states,new_inputs);
                
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
                while(learning_rate > low_learning_rate && armijo_flag == 0)
                    % Compute forward pass
                    [new_states,new_inputs]=forwards_pass(self,learning_rate);
                    new_cost = self.compute_cost(new_states,new_inputs);
                    % Calculate armijo condition
                    cost_difference = (current_cost - new_cost);
                    expected_cost_redu = learning_rate*self.expected_cost_redu_grad_ + learning_rate^2*self.expected_cost_redu_hess_;
                    armijo_flag = cost_difference/expected_cost_redu > armijo_threshold;
                    
                    if(armijo_flag == 1)
                        % Accept the new trajectory if armijo condition is
                        % met
                        current_cost = new_cost;
                        self.states_ = new_states;
                        self.inputs_ = new_inputs;
                    else
                        % If no improvement, decrease the learning rate
                        learning_rate = learning_speed*learning_rate;
                        %                         disp(['Reducing learning rate to: ',num2str(learning_rate)]);
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
                % Get the actual input and state
                current_x = states(ii,:)';
                current_u = inputs(ii,:)';
                
                adjustment_idx = self.current_idx_ -1;
                % Get the current target input and state
                current_x_des = self.target_states_(ii+adjustment_idx,:)';
                current_u_des = self.target_inputs_(ii+adjustment_idx,:)';
                
                % Compute differences from the target
                x_difference = current_x_des - current_x;
                u_difference = current_u_des - current_u;
                current_cost = u_difference'*self.R_k_*u_difference + x_difference'*self.Q_k_*x_difference;
                total_cost = total_cost+current_cost;
            end
            % Compute terminal cost
            terminal_x_diff = self.target_states_(self.current_idx_+self.n_timesteps_,:)'-states(end,:)';
            terminal_cost = terminal_x_diff'*self.Q_T_*terminal_x_diff;
            total_cost = total_cost+terminal_cost;
        end
        function [states,inputs] = rollout(self)
            states = zeros(self.n_timesteps_+1,self.n_states_);
            inputs = zeros(self.n_timesteps_,self.n_inputs_);
            current_state = self.current_state_;
            
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
        function [k_trj,K_trj,expected_cost_redu] = backwards_pass(self,states,inputs)
            % Initialize feedforward gains
            k_trj = zeros(self.n_timesteps_,self.n_inputs_);
            K_trj = zeros(self.n_timesteps_,self.n_inputs_,self.n_states_);
            % Initialize expected cost reduction
            expected_cost_redu = 0;
            expected_cost_redu_grad = 0;
            expected_cost_redu_hess = 0;
            
            idx_adjustment = self.current_idx_-1;
            % Iitialize gradient and hessian of the value function
            V_x = self.Q_T_*(states(end,:)'-self.target_states_(self.current_idx_+self.n_timesteps_,:)');
            V_xx = self.Q_T_;
            
            for ii = flip(1:self.n_timesteps_) % Go backwards in time
                % Get the current state and input
                current_x = states(ii,:)';
                current_u = inputs(ii,:)';
                
                current_x_des = self.target_states_(ii+idx_adjustment,:)';
                current_u_des = self.target_inputs_(ii+idx_adjustment,:)';
                
                % Get the gradient and hessian of the current cost
%                 l_x = self.Q_k_*(current_x_des-current_x); % Defined as zero right now because there is no desired trajectory
                l_x = self.Q_k_*(current_x-current_x_des); % Defined as zero right now because there is no desired trajectory
                l_xx = self.Q_k_; % Q_k should also be zero here
%                 l_u = self.R_k_*(current_u_des-current_u)+self.R_k_reg_*current_u;
                l_u = self.R_k_*(current_u-current_u_des);
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
                
                % Get the current expected cost reduction from each source
                current_cost_reduction_grad = -Q_u'*k;
                current_cost_reduction_hess = 0.5 * k'*(Q_uu)*k;
                current_cost_reduction = current_cost_reduction_grad + current_cost_reduction_hess;
                
                % Store each component separately for computing armijo
                % condition
                expected_cost_redu_grad = expected_cost_redu_grad + current_cost_reduction_grad;
                expected_cost_redu_hess = expected_cost_redu_hess + current_cost_reduction_hess;
                expected_cost_redu = expected_cost_redu+current_cost_reduction;
            end
            % Store expected cost reductions
            self.expected_cost_redu_grad_ = expected_cost_redu_grad;
            self.expected_cost_redu_hess_ = expected_cost_redu_hess;
            self.expected_cost_redu_ = expected_cost_redu;
            
            % Store gain schedule
            self.k_feedforward_= k_trj;
            self.K_feedback_ = K_trj;
        end
        function [states,inputs]=forwards_pass(self,learning_rate)
            states = zeros(self.n_timesteps_+1,self.n_states_);
            inputs = zeros(self.n_timesteps_,self.n_inputs_);
            current_state = self.current_state_;
            
            states(1,:) = current_state;
            
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

