close all; clear all;
symbolic_dynamics_acrobot();
% Initialize timings
dt = 0.001; % Need finer integration resolution
start_time = 0;
end_time = 4;
time_span = start_time:dt:end_time;

% Set desired state
n_states = 4;
n_inputs = 1;
init_state = [-pi/2;0;0;0]; % Define the initial state to be the origin 
target_state = [pi/2;0;0;0]; % Swing up and end with zero velocities

% Seed with random inputs centered about 0
rng(1);
initial_guess = 0.1*randn(size(time_span,2),n_inputs); 

% Define weighting matrices
Q_k = zeros(n_states,n_states); % zero weight to penalties along a trajectory since we are finding a trajectory
R_k = 0.005*eye(n_inputs);

Q_T = 10*eye(n_states);
Q_T(1,1) = 2000;
Q_T(2,2) = 1000;

% Parameters of the cart pole
mass1 = 1;
length1 = 1;
mass2 = 2; % more mass to swing better
length2 = 1;
gravity = 9.8;

parameters = [mass1,length1,mass2,length2,gravity];

% Specify max number of iterations
n_iterations = 50;

% Construct ilqr object
ilqr_ = ilqr(init_state,target_state,initial_guess,dt,start_time,end_time,@calc_f_disc,@calc_A_disc,@calc_B_disc,Q_k,R_k,Q_T,parameters,n_iterations);
% Solve
[states,inputs,k_feedforward,K_feedback,final_cost] = ilqr_.solve();

save('acrobot-swingup-trajectory.mat','states','inputs','dt','parameters','K_feedback','target_state');

figure(1);
plot(ilqr_.states_(:,1),ilqr_.states_(:,2));
hold on
plot(target_state(1),target_state(2),'ro');

figure(2);
plot(time_span,inputs);

figure(3);
animate_acrobot(states, dt)
% animate_cartpole(states,inputs, dt)

