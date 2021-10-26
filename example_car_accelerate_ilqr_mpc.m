close all; clear all;
% Create dynamic functions
symbolic_dynamics_car();

% Load in trajectory for swing up
data = load('car-accelerate-trajectory.mat');
states = data.states; inputs = data.inputs; dt = data.dt; parameters = data.parameters; target_state = data.target_state;
duration = size(inputs,1);

% Define weighting matrices
n_states = size(states,2); n_inputs = size(inputs,2);
Q_k = 0.5*eye(n_states); % We care most about reaching the end goal of swinging up
R_k = 0.1*eye(n_inputs);

% Weight position more than velocity because velocities are generally
% bigger
Q_T = 100*eye(n_states);
% Q_T(2,2) = 10;

% Set the mpc horizon
horizon = 200;
% Set max numebr of iterations
n_iterations = 10;

% Pad the target states and input with the size of the horizon (assume the
% target state is a fixed point)
states = [states;repmat(target_state',horizon,1)];
inputs = [inputs;repmat(zeros(1,n_inputs),horizon,1)];

ilqr_mpc_ = ilqr_mpc(states,inputs,dt,horizon,@calc_f_disc,@calc_A_disc,@calc_B_disc,Q_k,R_k,Q_T,parameters,n_iterations);

% Simulate with a perturbation
new_states = zeros(duration+1,size(states,2));
new_inputs= zeros(duration,size(inputs,2));
init_state = states(1,:)'; % Get the initial state
% current_state = init_state + [0;-pi/12;0;0]; % Add larger perturbation
current_state = init_state + [0;1;0;0]; % Add larger perturbation
new_states(1,:) = current_state';

% initalize storage for mpc solutions
mpc_states = zeros(duration,horizon+1,size(states,2));
mpc_inputs = zeros(duration,horizon,size(inputs,2));

for ii=1:duration
    % Get the current gains and compute the feedforward and
    % feedback terms
    [states_solve,inputs_solve,k_feedforward,K_feedback,current_cost] = ilqr_mpc_.solve_ilqr(ii,current_state);
    
    % Store mpc solve
    mpc_states(ii,:,:) = states_solve;
    mpc_inputs(ii,:,:) = inputs_solve;
    
    % Take the first input of the optimal trajectory
    current_input = inputs_solve(1,:)';
    
    % simualte forward
    next_state = calc_f_disc(current_state,current_input,dt,parameters);
    
    % Store states and inputs
    new_states(ii+1,:) = next_state';
    new_inputs(ii,:) = current_input';
    
    % Update the current state
    current_state = next_state;
    figure(4);
    h1 = plot(states(:,1),states(:,2),'k-');
    hold on
    
    h3 = plot(states(ii:(ii+horizon-1),1),states(ii:(ii+horizon-1),2),'ro');
    h2 = plot(new_states(:,1),new_states(:,2),'b.');
    h4 = plot(states_solve(:,1),states_solve(:,2),'b--');
    legend([h1,h2,h3,h4],"Reference Trajectory","Perturbed Trajectory","Reference Horizon","MPC Horizon");
    xlabel('$$x$$');
    ylabel('$$y$$');
    hold off
    
end

figure(1);
h1 = plot(states(:,1),states(:,2),'k-');
hold on
h2 = plot(new_states(:,1),new_states(:,2),'b--');
legend([h1,h2],"Reference Trajectory","Perturbed Trajectory");
title("State trajectory phase plot");
xlabel('$$x$$');
ylabel('$$y$$');

figure(2);
h1 = plot(inputs(:,1),'k-');
hold on
h2 = plot(new_inputs(:,1),'b--');
legend([h1,h2],"Reference Trajectory","Perturbed Trajectory");
title("Acceleration command time series");
xlabel('$$k$$');
ylabel('$$v$$');

figure(3);
h1 = plot(inputs(:,2),'k-');
hold on
h2 = plot(new_inputs(:,2),'b--');
legend([h1,h2],"Reference Trajectory","Perturbed Trajectory");
title("Steering velocity command time series");
xlabel('$$k$$');
ylabel('$$\dot{\theta}$$');

% figure(3);
% animate_cartpole(new_states,new_inputs, dt)
% figure(3);
% animate_pendulum_mpc_tracking(new_states,states,dt,mpc_states)

