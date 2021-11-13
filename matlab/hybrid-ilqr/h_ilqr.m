classdef h_ilqr < handle
    % This class sets up the optimization problem with the constructor and
    % solve will return the optimal set of inputs and gains to achieve a desired target state.
    % Note that the discrete dynamics of the system and it's Jacobians are
    % required to be functions ready to be called with
    % inputs(state,input,dt,parameters)
    properties
        init_mode_
        init_state_
        states_
        target_state_
        inputs_
        modes_
        dt_
        start_time_
        end_time_
        time_span_
        A_
        B_
        Q_k_
        R_k_
        Q_T_
        n_hybrid_modes_
        parameters_
        K_feedback_
        k_feedforward_
        n_states_
        n_inputs_
        f_
        r_
        g_
        salts_
        n_timesteps_
        n_iterations_
        
        % Hybrid storage
        trajectory_struct_
        impact_states_
        reset_states_ 
        impact_idx_vec_
        reset_mode_vec_ 
        impact_diff_time_vec_
        reset_diff_time_vec_
        transition_inputs_
        impact_mode_vec_
            
        % Expected cost reductions
        expected_cost_redu_
        expected_cost_redu_grad_
        expected_cost_redu_hess_
        
        % Optimality condition
        min_reduction_
        
    end
    methods
        % Constructor
%         function self = h_ilqr(init_state,init_mode,target_state,initial_guess,dt,start_time,end_time,f,resets,guards,salts,A,B,Q_k,R_k,Q_T,parameters,n_iterations)
          function self = h_ilqr(optimization_problem_struct,dynamic_struct)
            % Optimization problem
            
            % Dynamics
            
            self.init_mode_ = optimization_problem_struct.init_mode;
            self.init_state_ = optimization_problem_struct.init_state;
            self.target_state_ = optimization_problem_struct.target_state;
            self.inputs_ = optimization_problem_struct.initial_guess;
            self.n_states_ = size(optimization_problem_struct.init_state,1);
            self.n_inputs_ = size(optimization_problem_struct.initial_guess,2);
            
            self.dt_ = optimization_problem_struct.dt;
            self.start_time_ = optimization_problem_struct.start_time;
            self.end_time_ = optimization_problem_struct.end_time;
            self.time_span_ = optimization_problem_struct.start_time:optimization_problem_struct.dt:optimization_problem_struct.end_time;
            self.n_timesteps_ = size(self.time_span_,2);
            % Dynamics
            self.f_ = dynamic_struct.f;
            self.A_ = dynamic_struct.A_disc;
            self.B_ = dynamic_struct.B_disc;
            
            % Hybrid dynamics
            self.r_ = dynamic_struct.resets;
            self.g_ = dynamic_struct.guards;
            self.salts_ = dynamic_struct.salts;
            
            % Weighting
            self.Q_k_ = optimization_problem_struct.Q_k;
            self.R_k_ = optimization_problem_struct.R_k;
            self.Q_T_ = optimization_problem_struct.Q_T;
            self.parameters_ = dynamic_struct.parameters;
            
            % Max iterations
            self.n_iterations_ = optimization_problem_struct.n_iterations;
            
            % Optimality condition
            self.min_reduction_ = optimization_problem_struct.min_reduction;
        end
        function [states,inputs,modes,trajectory_struct,k_feedforward,K_feedback,current_cost,expected_reduction] = solve(self)
            % Compute the rollout to get the initial trajectory with the
            % initial guess
            [states,inputs] = self.rollout();
%             figure(3);
% animate_ball_drop_circle(states,self.dt_)
            % Compute the current cost of the initial trajectory
            current_cost = self.compute_cost(states,inputs);
            
            learning_speed = 0.95; % This can be modified, 0.95 is very slow
            low_learning_rate = 0.05; % if learning rate drops to this value stop the optimization
%             low_expected_reduction = 1e-2; % Determines optimality
            armijo_threshold = 0.1; % Determines if current line search solve is good (this is typically labeled as "c")
            for ii = 1:self.n_iterations_
                disp(['Starting iteration: ',num2str(ii)]);
                % Compute the backwards pass
                [k_feedforward,K_feedback,expected_reduction] = self.backwards_pass();
                
                if(abs(expected_reduction)<self.min_reduction_)
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
                    [new_states,new_inputs,new_modes,new_trajectory_struct]=forwards_pass(self,learning_rate);
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
                        self.modes_ = new_modes;
                        self.trajectory_struct_ = new_trajectory_struct;
                        % Store new hybrid trajectory info
                        self.impact_states_ = new_trajectory_struct.impact_states_;
                        self.reset_states_ = new_trajectory_struct.reset_states_;
                        self.impact_idx_vec_ = new_trajectory_struct.impact_idx_vec_;
                        self.reset_mode_vec_ = new_trajectory_struct.reset_mode_vec_;
                        self.impact_diff_time_vec_ = new_trajectory_struct.impact_diff_time_vec_;
                        self.reset_diff_time_vec_ = new_trajectory_struct.reset_diff_time_vec_;
                        self.transition_inputs_ = new_trajectory_struct.transition_inputs_;
                        self.impact_mode_vec_ = new_trajectory_struct.impact_mode_vec_;
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
            modes = self.modes_;
            trajectory_struct = self.trajectory_struct_;
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
            dt = self.dt_;
            
            % initialize hybrid
            modes = zeros(self.n_timesteps_);
            impact_states = [];
            reset_states = [];
            impact_idx_vec = [];
            reset_mode_vec = [];
            impact_diff_time_vec = [];
            reset_diff_time_vec = [];
            transition_inputs = [];
            impact_mode_vec = [];
            
            current_state = self.init_state_;
            current_mode = self.init_mode_;
            states(1,:) = current_state';
            
            
            for ii=1:self.n_timesteps_
                current_input = self.inputs_(ii,:)';
                if(current_mode ==2)
                   disp(''); 
                end
                % simulate each timestep with ode event checking, we can
                % make speed ups by using discrete dynamics when far away
                % from the guard ||g||<1 ?
                options = odeset('Events', @(t,x)guardFunctions(t,x,current_input,self.g_{current_mode},self.parameters_),'MaxStep',0.01);
                tspan = self.dt_*(ii - 1):self.dt_/10:self.dt_*ii;
                [t,x,te,xe,ie] = ode45(@(t,x)dynamics(t,x,self.f_{current_mode},current_input,self.parameters_),tspan,current_state,options);
                %               [t,x,te,xe,ie] = ode45(@(t,x)dynamics_input(t,x,current_input,params,current_mode),tspan,current_state,options);
                next_state = x(end,:)';
                while(~isempty(te))
                    %         disp('');
                    impact_states = [impact_states;next_state'];
                    next_state = self.r_{current_mode}(next_state,current_input,self.parameters_);
                    reset_states = [reset_states;next_state'];
                    dt1 = t(end) - dt*(ii - 1);
                    dt2 = dt*ii - t(end);
                    impact_mode_vec = [impact_mode_vec;current_mode];
                    dt1 = t(end)-dt*(ii-1);
                    dt2 = dt*ii - t(end);
                    tspan = linspace(dt*(ii - 1)+dt2,dt*ii,10);
                    if(current_mode == 1)
                        current_mode = 2;
                    else
                        current_mode = 1;
                    end
                    options = odeset('Events', @(t,x)guardFunctions(t,x,current_input,self.g_{current_mode},self.parameters_),'MaxStep',0.01);
                    [t,x,te,xe,ie] = ode45(@(t,x)dynamics(t,x,self.f_{current_mode},current_input,self.parameters_),tspan,current_state,options);
                    impact_idx_vec = [impact_idx_vec;ii];
                    reset_mode_vec = [reset_mode_vec;current_mode];
                    impact_diff_time_vec = [impact_diff_time_vec;dt1];
                    reset_diff_time_vec = [reset_diff_time_vec;dt2];
                    transition_inputs = [transition_inputs;current_input];
                end
                
                
                %                 next_state = self.f_(current_state,current_input,self.dt_,self.parameters_);
                % If we hit a guard apply reset
                
                % Store states and inputs
                states(ii+1,:) = next_state';
                inputs(ii,:) = current_input'; % in case we have a control law, we store the input used
                modes(ii) = current_mode;
                % Update the current state
                current_state = next_state;
            end
            % Store the trajectory (states,inputs)
            %             animate_bouncing_ball(states,dt,inputs)
            self.states_ = states;
            self.inputs_= inputs;
            
            % Store hybrid
            self.modes_ = modes;
            self.impact_states_ = impact_states;
            self.reset_states_ = reset_states;
            self.impact_idx_vec_ = impact_idx_vec;
            self.reset_mode_vec_ = reset_mode_vec;
            self.impact_diff_time_vec_ = impact_diff_time_vec;
            self.reset_diff_time_vec_ = reset_diff_time_vec;
            self.transition_inputs_ = transition_inputs;
            self.impact_mode_vec_ = impact_mode_vec;
        end
        function [k_trj,K_trj,expected_cost_redu] = backwards_pass(self)
            % Initialize feedforward gains
            k_trj = zeros(size(self.inputs_));
            K_trj = zeros(size(self.inputs_,1),size(self.inputs_,2),size(self.states_,2));
            % Initialize expected cost reduction
            expected_cost_redu = 0;
            expected_cost_redu_grad = 0;
            expected_cost_redu_hess = 0;
            
            % Iitialize gradient and hessian of the value function
            V_x = self.Q_T_*(self.states_(end,:)'-self.target_state_);
            V_xx = self.Q_T_;
            
            for ii = flip(1:self.n_timesteps_) % Go backwards in time
                                % Get the current state and input
                current_x = self.states_(ii,:)';
                current_u = self.inputs_(ii,:)';
                current_mode = self.modes_(ii);
                
                % If impact occured, then apply saltation matrix
                if(any(ii==self.impact_idx_vec_))
                    current_impact_idx = find(self.impact_idx_vec_==ii);
                    impact_states = self.impact_states_(current_impact_idx,:)';
                    impact_inputs = self.transition_inputs_(current_impact_idx,:)';
                    impact_mode = self.impact_mode_vec_(current_impact_idx);
                    salt = self.salts_{impact_mode}(impact_states,impact_inputs,self.parameters_);
                    % Update with saltation matrix
                    V_x = salt'*V_x;
                    V_xx = salt'*V_xx*salt;
                    
                    % Only assume a single hybrid mode
                    current_mode= impact_mode;
                end
                

                % Get the gradient and hessian of the current cost
                l_x = zeros(size(current_x)); % Defined as zero right now because there is no desired trajectory
                l_xx = self.Q_k_; % Q_k should also be zero here
                l_u = self.R_k_*current_u;
                l_uu = self.R_k_;
                
                % Get the jacobian of the discretized dynamics
                A_k = self.A_{current_mode}(current_x,current_u,self.dt_,self.parameters_);
                B_k =self.B_{current_mode}(current_x,current_u,self.dt_,self.parameters_);
                
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
        function [states,inputs,modes,trajectory_struct]=forwards_pass(self,learning_rate)
            states = zeros(self.n_timesteps_+1,self.n_states_);
            inputs = zeros(self.n_timesteps_,self.n_inputs_);
            current_state = self.init_state_;
            dt = self.dt_;
            
             % initialize hybrid
            modes = zeros(self.n_timesteps_);
            impact_states = [];
            reset_states = [];
            impact_idx_vec = [];
            reset_mode_vec = [];
            impact_diff_time_vec = [];
            reset_diff_time_vec = [];
            transition_inputs = [];
            impact_mode_vec = [];
            % Count how many hybrid transitions are made
            hybrid_transitions = 0;
            
            % set the first state to be the initial
            states(1,:) = current_state;
            current_mode = self.init_mode_;
            for ii=1:self.n_timesteps_
                % Get the current gains and compute the feedforward and
                % feedback terms
                reference_hybrid_transitions = sum(ii>=self.impact_idx_vec_); %This seems right?
                mode_count_difference = hybrid_transitions-reference_hybrid_transitions;
                
                % Check if mode mismatch
                if(current_mode ~=self.modes_(ii))
                   if(mode_count_difference<0)
                       % Late impact
                       impact_idx = self.impact_idx_vec_(reference_hybrid_transitions);
                       ref_state = self.impact_states_(reference_hybrid_transitions,:)';
                       ref_input = self.transition_inputs_(reference_hybrid_transitions,:)';
                       ref_k_feedforward = self.k_feedforward_(impact_idx,:)';
                       ref_K_feedback = reshape(self.K_feedback_(impact_idx,:,:),self.n_inputs_,self.n_states_);
                   end
                   if(mode_count_difference>0)
                       % early impact
                       if(hybrid_transitions>size(self.impact_idx_vec_,1))
                           % We don't have a reference for this new domain
                           ref_state = self.states_(end,:)';
                           ref_input = self.inputs_(end,:)';
                           ref_k_feedforward = 0*self.inputs_(end,:)'; % Use no feedforward
                           ref_K_feedback = reshape(self.K_feedback_(end,:,:),self.n_inputs_,self.n_states_); % Use the last gain scheduled, might not be good
                       else
                           impact_idx = self.impact_idx_vec_(hybrid_transitions);
                           ref_state = self.reset_states_(hybrid_transitions,:)';
                           ref_input = self.transition_inputs_(hybrid_transitions,:)';
                           ref_k_feedforward = self.k_feedforward_(impact_idx,:)';
                           ref_K_feedback = reshape(self.K_feedback_(impact_idx,:,:),self.n_inputs_,self.n_states_);
                       end
                   end
                   % Hold pre impact state, inputs, and gains constant
                   % Set gain
                   % Set feedforward
                   % Set reference state
                    current_feedforward = learning_rate*ref_k_feedforward;
                    current_feedback = ref_K_feedback*(current_state-ref_state);
                    current_input = ref_input + current_feedback + current_feedforward;
                else
                    current_feedforward = learning_rate*self.k_feedforward_(ii,:)';
                    current_feedback = reshape(self.K_feedback_(ii,:,:),self.n_inputs_,self.n_states_)*(current_state-self.states_(ii,:)');
                    current_input = self.inputs_(ii,:)' + current_feedback + current_feedforward;
                end
                % simualte forward
                % simulate each timestep with ode event checking, we can
                % make speed ups by using discrete dynamics when far away
                % from the guard ||g||<1 ?
                options = odeset('Events', @(t,x)guardFunctions(t,x,current_input,self.g_{current_mode},self.parameters_),'MaxStep',0.01);
                tspan = self.dt_*(ii - 1):self.dt_/5:self.dt_*ii;
                [t,x,te,xe,ie] = ode45(@(t,x)dynamics(t,x,self.f_{current_mode},current_input,self.parameters_),tspan,current_state,options);
                %               [t,x,te,xe,ie] = ode45(@(t,x)dynamics_input(t,x,current_input,params,current_mode),tspan,current_state,options);
                next_state = x(end,:)';
                while(~isempty(te))
                    %         disp('');
                    impact_states = [impact_states;next_state'];
                    next_state = self.r_{current_mode}(next_state,current_input,self.parameters_);
                    reset_states = [reset_states;next_state'];
                    dt1 = t(end) - dt*(ii - 1);
                    dt2 = dt*ii - t(end);
                    impact_mode_vec = [impact_mode_vec;current_mode];
                    dt1 = t(end)-dt*(ii-1);
                    dt2 = dt*ii - t(end);
                    tspan = linspace(dt*(ii - 1)+dt2,dt*ii,5);
                    if(current_mode == 1)
                        current_mode = 2;
                    else
                        current_mode = 1;
                    end
                    options = odeset('Events', @(t,x)guardFunctions(t,x,current_input,self.g_{current_mode},self.parameters_),'MaxStep',0.01);
                    [t,x,te,xe,ie] = ode45(@(t,x)dynamics(t,x,self.f_{current_mode},current_input,self.parameters_),tspan,current_state,options);
                    impact_idx_vec = [impact_idx_vec;ii];
                    reset_mode_vec = [reset_mode_vec;current_mode];
                    impact_diff_time_vec = [impact_diff_time_vec;dt1];
                    reset_diff_time_vec = [reset_diff_time_vec;dt2];
                    transition_inputs = [transition_inputs;current_input];
                    hybrid_transitions = hybrid_transitions + 1;
                end
%                 next_state = self.f_(current_state,current_input,self.dt_,self.parameters_);
                % Store states and inputs
                states(ii+1,:) = next_state';
                inputs(ii,:) = current_input';
                modes(ii) = current_mode;
                % Update the current state
                current_state = next_state;
            end
            trajectory_struct.impact_states_ = impact_states;
            trajectory_struct.reset_states_ = reset_states;
            trajectory_struct.impact_idx_vec_ = impact_idx_vec;
            trajectory_struct.reset_mode_vec_ = reset_mode_vec;
            trajectory_struct.impact_diff_time_vec_ = impact_diff_time_vec;
            trajectory_struct.reset_diff_time_vec_ = reset_diff_time_vec;
            trajectory_struct.transition_inputs_ = transition_inputs;
            trajectory_struct.impact_mode_vec_ = impact_mode_vec;
%             animate_bouncing_ball(states,dt,inputs)
        end
    end
end

