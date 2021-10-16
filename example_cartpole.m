close all; clear all;
symbolic_dynamics_cartpole();
% Initialize timings
dt = 0.005;
start_time = 0;
end_time = 5;
time_span = start_time:dt:end_time;

% Set desired state
n_states = 4;
n_inputs = 1;
init_state = [0;0;0;0]; % Define the initial state to be the origin 
target_state = [0;pi;0;0]; % Swing the pole up and end at the x origin with zero velocities

% Seed with random inputs centered about 0
rng(1);
initial_guess = 0.5*randn(size(time_span,2),n_inputs); 

% Define weighting matrices
Q_k = zeros(n_states,n_states); % zero weight to penalties along a trajectory since we are finding a trajectory
R_k = 0.0001*eye(n_inputs);

Q_T = 1*eye(n_states);
Q_T(2,2) = 100;

% Parameters of the cart pole
cart_mass = 1;
pole_mass = 1;
gravity = 9.8;
pole_length = 1;
parameters = [cart_mass,pole_mass,pole_length,gravity];

% Specify max number of iterations
n_iterations = 50;

% Construct ilqr object
ilqr_ = ilqr(init_state,target_state,initial_guess,dt,start_time,end_time,@calc_f_disc,@calc_A_disc,@calc_B_disc,Q_k,R_k,Q_T,parameters,n_iterations);
% Solve
[states,inputs,k_feedforward,K_feedback,final_cost] = ilqr_.solve();

save('cartpole-swingup-trajectory.mat','states','inputs','dt','parameters','K_feedback','target_state');

figure(1);
plot(ilqr_.states_(:,2),ilqr_.states_(:,4));
hold on
plot(target_state(2),target_state(4),'ro');

figure(2);
plot(inputs);

figure(3);
animate_cartpole(states,inputs, dt)

