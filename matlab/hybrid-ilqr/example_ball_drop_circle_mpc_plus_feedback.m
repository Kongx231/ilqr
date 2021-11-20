close all; clear all;
% Compute symbolic dynamics and create functions
% symbolic_dynamics_plastic_impact_circle_drop();
% load target trajectory
data = load('ball-drop-circle-trajectory.mat');
states = data.states; inputs = data.inputs; modes = data.modes; dt = data.dt; parameters = data.parameters; trajectory_struct = data.trajectory_struct;
K_feedback = data.K_feedback;
duration = size(inputs,1);

% Initialize timings
start_time = 0;
end_time = 1;
time_span = start_time:dt:end_time;

% Set desired state
n_states = 4;
n_inputs = 2;
init_state = states(1,:)'; % Define the initial state to be the origin with no velocity
init_mode = modes(1);
target_states = states;
target_inputs = inputs;
target_modes = modes;
target_trajectory_struct = trajectory_struct;
target_K_feedback = K_feedback;

% Initial guess of zeros, but you can change it to any guess
initial_guess = target_inputs;
% Define weighting matrices
Q_k = 0.01*eye(n_states); % zero weight to penalties along a strajectory since we are finding a trajectory
R_k = 0.01*eye(n_inputs);

% Set the terminal cost
Q_T = 100*eye(n_states);

% Set the mpc horizon
horizon = 150;
% Set max numebr of iterations
n_iterations = 10;

% Specify optimality condition
min_reduction = 1e-3;

% Define optimization struct
optimization_struct.init_state = init_state;
optimization_struct.init_mode = init_mode;

optimization_struct.target_states = [target_states;repmat(target_states(end,:),horizon,1)];
optimization_struct.target_inputs = [target_inputs;repmat(target_inputs(end,:),horizon,1)];
optimization_struct.target_modes = [target_modes;repmat(target_modes(end),horizon,1)];
optimization_struct.target_K_feedback = [target_K_feedback;repmat(target_K_feedback(end,:,:),horizon,1)];

optimization_struct.target_trajectory_struct = target_trajectory_struct;

optimization_struct.dt = dt;
optimization_struct.start_time = start_time;
optimization_struct.end_time = end_time;
optimization_struct.n_iterations = n_iterations;
optimization_struct.min_reduction = min_reduction;
optimization_struct.Q_T = Q_T;
optimization_struct.Q_k = Q_k;
optimization_struct.R_k = R_k;
optimization_struct.horizon = horizon;

% Construct ilqr object
A_disc = {@calc_A_lin1_disc,@calc_A_lin2_disc};
B_disc = {@calc_B_lin1_disc,@calc_B_lin2_disc};
f = {@calc_f1,@calc_f2};
resets = {@calc_r12,@calc_r21};
guards = {@calc_g12,@calc_g21};
guard_jacobians = {@calc_Dg12,@calc_Dg21};
salts = {@calc_salt12,@calc_salt21};

% Define dynamics struct
dynamics_struct.A_disc = A_disc;
dynamics_struct.B_disc = B_disc;
dynamics_struct.f = f;
dynamics_struct.resets = resets;
dynamics_struct.guards = guards;
dynamics_struct.salts = salts;
dynamics_struct.parameters = parameters;
dynamics_struct.guard_jacobians = guard_jacobians;

ilqr_mpc_ = h_ilqr_mpc(optimization_struct,dynamics_struct);
% Solve
% [states,inputs,modes,trajectory_struct,k_feedforward,K_feedback,final_cost,expected_reduction] = ilqr_.solve();
% save('ball-drop-circle-trajectory.mat','states','inputs','modes','trajectory_struct','dt','parameters','k_feedforward','K_feedback'); % Save trajectory to track later

% Simulate with a perturbation
new_states = zeros(duration+1,size(states,2));
new_inputs= zeros(duration,size(inputs,2));
exit_flags = zeros(duration,1);
new_modes = zeros(duration,1);

init_state = states(1,:)'; % Get the initial state
% current_state = init_state + [-0.5;1;0;0]; % Add larger perturbation
current_state = init_state + [-0.1;0.1;0;0]; % Add larger perturbation
% current_state = init_state + [0.5;-1;0;0]; % Add larger perturbation
% current_state = init_state+ [0.1;-0.1;0;0]; % Add larger perturbation
current_mode = init_mode;
new_states(1,:) = current_state';

% initalize storage for mpc solutions
mpc_states = zeros(duration,horizon+1,size(states,2));
mpc_inputs = zeros(duration,horizon,size(inputs,2));

solve_interval = 50;
mpc_counter = solve_interval;

% Record ball drop circle
bRecord = 1;
if bRecord
    Filename = 'ball_drop_circle_periodic_solve_smaller_disturbance_late_impact';
    v = VideoWriter(Filename, 'MPEG-4');
    myVideo.Quality = 100;
    open(v);
end

for ii=1:duration
    % Get the current gains and compute the feedforward and
    % feedback terms
    if(mpc_counter>=solve_interval)
        % Should have current trajectory struct
        [states_solve,inputs_solve,modes_solve,trajectory_struct_solve,k_feedforward,K_feedback,current_cost,expected_reduction_solve,exit_flag] = ilqr_mpc_.solve_ilqr(ii,current_state,current_mode);
        % Store mpc solve
        mpc_states(ii,:,:) = states_solve;
        mpc_inputs(ii,:,:) = inputs_solve;
        
        % reset the counter
        mpc_counter = 0;
    end
    % Update the counter
    mpc_counter = mpc_counter+1;
    
    %     % Take the first input of the optimal trajectory
    %     current_input = inputs_solve(1,:)';
    
    % TODO Check if mode mismatch
    
    % Use current optimal trajectory to track back towards the reference
    current_error = current_state - states_solve(mpc_counter,:)';
    current_feedback_gain = reshape(K_feedback(mpc_counter,:,:),n_inputs,n_states);
    current_input = inputs_solve(mpc_counter,:)' + current_feedback_gain*current_error;
    
    tspan = dt*(ii - 1):dt/10:dt*ii;
    % simualte forward TODO: Create simulate function for hybrid iLQR
    [next_state,current_mode,hybrid_timestep_struct] =ilqr_mpc_.simulate_hybrid_timestep(current_state,current_input,current_mode,tspan,ii);
    %     next_state = calc_f_disc(current_state,current_input,dt,parameters);
    
    % Store states and inputs
    new_states(ii+1,:) = next_state';
    new_inputs(ii,:) = current_input';
    new_modes(ii) = current_mode;
    
    % Store exit flag
    exit_flags(ii) = exit_flag;
    
    % Update the current state
    current_state = next_state;
    
        if(exit_flag == 1)
        title(['Timestep: ',num2str(ii),', Converged']);
    else
        title(['Timestep: ',num2str(ii),', Not Converged']);
    end

    figure(4);
    draw_circle_constraint()
    h1 = plot(optimization_struct.target_states(:,1),optimization_struct.target_states(:,2),'k-');
    hold on
    
    h3 = plot(ilqr_mpc_.target_states_(ii:(ii+horizon-1),1),ilqr_mpc_.target_states_(ii:(ii+horizon-1),2),'ro');
    h2 = plot(new_states(1:ii,1),new_states(1:ii,2),'b.');
    h4 = plot(states_solve(:,1),states_solve(:,2),'b--');
    if(exit_flag == 1)
        title(['Timestep: ',num2str(ii),', Converged']);
    else
        title(['Timestep: ',num2str(ii),', Not Converged']);
    end
    %     legend([h1,h2,h3,h4],"Reference Trajectory","Perturbed Trajectory","Reference Horizon","MPC Horizon");
    %     xlabel('$$\theta$$');
    %     ylabel('$$\dot{\theta}$$');
    hold off
    
    %     figure(5);
    %     hold on
    %     plot(inputs_solve);
    %     pause(0.1);
    drawnow limitrate
    if bRecord
        frame = getframe(gcf);
        writeVideo(v,frame);
    end
end
if bRecord
    close(v);
end


bad_solves = find(exit_flags == -1);
if(isempty(bad_solves))
    bad_solves = 1;
end

contact_solves = find(modes == -1);
new_impact_idx = find(diff(new_modes)~=0)+1;
old_impact_idx = trajectory_struct.impact_idx_vec_;

figure(1);
subplot(2,2,1)
h2 = plot(target_states(:,1),'k');
hold on
h1 = plot(new_states(:,1),'b--');
h3 = highlight(bad_solves(1),bad_solves(end));
h4 = vline(old_impact_idx,'k--');
h5 = vline(new_impact_idx,'b--');
ylabel('$${x}$$');
subplot(2,2,2)
h2 = plot(target_states(:,3),'k');
hold on
h1 = plot(new_states(:,3),'b--');
h3 = highlight(bad_solves(1),bad_solves(end));
h4 = vline(old_impact_idx,'k--');
h5 = vline(new_impact_idx,'b--');
ylabel('$$\dot{x}$$');

subplot(2,2,3)
h2 = plot(target_states(:,2),'k');
hold on
h1 = plot(new_states(:,2),'b--');
h3 = highlight(bad_solves(1),bad_solves(end));
h4 = vline(old_impact_idx,'k--');
h5 = vline(new_impact_idx,'b--');
xlabel('Timestep');
ylabel('$${y}$$');

subplot(2,2,4)
h2 = plot(target_states(:,4),'k');
hold on
h1 = plot(new_states(:,4),'b--');
h3 = highlight(bad_solves(1),bad_solves(end));
h4 = vline(old_impact_idx,'k--');
h5 = vline(new_impact_idx,'b--');
% legend([h1,h2,h3],"Final Trajectory","Target Trajectory","Not Converged");
xlabel('Timestep');
ylabel('$$\dot{y}$$');

% figure(1);
% h2 = plot(target_states(:,1),target_states(:,2),'k');
% hold on
% h1 = plot(new_states(:,1),new_states(:,2),'b--');
% legend([h1,h2],"Final Trajectory","Target Trajectory");
% xlabel('$$x$$');
% ylabel('$$y$$');

figure(2);
subplot(2,1,1)
h2 = plot(target_inputs(:,1),'k');
hold on
h1 = plot(new_inputs(:,1),'b--');
h3 = highlight(bad_solves(1),bad_solves(end));
h4 = vline(old_impact_idx,'k--');
h5 = vline(new_impact_idx,'b--');
ylabel('$$u_x$$');
subplot(2,1,2)
h2 = plot(target_inputs(:,2),'k');
hold on
h1 = plot(new_inputs(:,2),'b--');
h3 = highlight(bad_solves(1),bad_solves(end));
h4 = vline(old_impact_idx,'k--');
h5 = vline(new_impact_idx,'b--');
legend([h1,h2,h3,h4,h5],"Final Trajectory","Target Trajectory","Not Converged","Old Impact","New Impact");
xlabel('Timestep');
ylabel('$$u_y$$');

% figure(3);
% animate_ball_drop_circle(states,dt,inputs)

