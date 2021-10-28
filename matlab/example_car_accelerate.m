close all; clear all;
% 
symbolic_dynamics_car();
% Initialize timings
dt = 0.005;
start_time = 0;
end_time = 4;
time_span = start_time:dt:end_time;

% Set desired state
n_states = 4;
n_inputs = 2;
init_state = [0;0;0;0]; % Define the initial state to be the origin with no velocity and straight heading
target_vel = 1;
time_elapsed = end_time-start_time;
target_state = [time_elapsed*target_vel;0;target_vel;0]; % Get to 3 meters to the right facing up and stop

% Set initial guess as zeros
initial_guess = [0*ones(size(time_span,2),1),0*ones(size(time_span,2),1)];

% Define weighting matrices
Q_k = zeros(n_states,n_states); % zero weight to penalties along a strajectory since we are finding a trajectory
R_k = 0.0001*eye(n_inputs);


Q_T = 100*eye(n_states);
% Q_T(1,1) = 100;
% Q_T(2,2) = 100;
% Q_T(3,3) = 100;

% There are no physical parameters to adjust
parameters = [];

% Specify max number of iterations
n_iterations = 50;

% Construct ilqr object
ilqr_ = ilqr(init_state,target_state,initial_guess,dt,start_time,end_time,@calc_f_disc,@calc_A_disc,@calc_B_disc,Q_k,R_k,Q_T,parameters,n_iterations);
% Solve
[states,inputs,k_feedforward,K_feedback,final_cost] = ilqr_.solve();

save('car-accelerate-trajectory.mat','states','inputs','dt','parameters','k_feedforward','K_feedback','target_state'); % Save trajectory to track later

figure(1);
plot(ilqr_.states_(:,1),ilqr_.states_(:,2));
hold on
plot(target_state(1),target_state(2),'ro');
xlabel('x');
ylabel('y');

figure(3);
plot(inputs(:,1))

% figure(2);
% animate_car(states)