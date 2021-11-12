close all; clear all;
% Compute symbolic dynamics and create functions
symbolic_dynamics_pendulum();

% Initialize timings
dt = 0.005;
start_time = 0;
end_time = 5;
time_span = start_time:dt:end_time;

% Set desired state
n_states = 2;
n_inputs = 1;
init_state = [0;0]; % Define the initial state to be the origin with no velocity
target_state = [pi;0]; % Swing pendulum upright

% Initial guess of zeros, but you can change it to any guess
initial_guess = 0.0*ones(size(time_span,2),1);
% Define weighting matrices
Q_k = zeros(n_states,n_states); % zero weight to penalties along a strajectory since we are finding a trajectory
R_k = 0.001*eye(n_inputs);

% Set the terminal cost
Q_T = 100*eye(n_states);

% Set the physical parameters of the system
mass = 1;
gravity = 9.8;
pendulum_length = 1;
parameters = [mass,gravity,pendulum_length];

% Specify max number of iterations
n_iterations = 50;

% Construct ilqr object
ilqr_ = ilqr(init_state,target_state,initial_guess,dt,start_time,end_time,@calc_f_disc,@calc_A_disc,@calc_B_disc,Q_k,R_k,Q_T,parameters,n_iterations);
% Solve
[states,inputs,k_feedforward,K_feedback,final_cost] = ilqr_.solve();
save('swingup-trajectory.mat','states','inputs','dt','parameters','k_feedforward','K_feedback'); % Save trajectory to track later

figure(1);
h1 = plot(states(:,1),states(:,2),'k');
hold on
h2 = plot(target_state(1),target_state(2),'ro');
legend([h1,h2],"Final Trajectory","Target State");
xlabel('$$\theta$$');
ylabel('$$\dot{\theta}$$');

figure(2);
animate_pendulum(states,dt)

