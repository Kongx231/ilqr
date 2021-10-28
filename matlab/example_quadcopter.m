close all; clear all;
symbolic_dynamics_quadcopter();
% Initialize timings
dt = 0.005;
start_time = 0;
end_time = 2;
time_span = start_time:dt:end_time;

% Set desired state
n_states = 12;
n_inputs = 4;
init_state = zeros(n_states,1); % Define the initial state to be the origin 
target_state = init_state; % Swing the pole up and end at the x origin with zero velocities
% go a unit distance in each position 
target_state(1)= 1;
target_state(2)= 1;
target_state(3)= 1;

% Define weighting matrices
Q_k = zeros(n_states,n_states); % zero weight to penalties along a trajectory since we are finding a trajectory
R_k = 0.1*eye(n_inputs);

Q_T = 100*eye(n_states);


% Generic parameters for quadcopter
J_x = 0.05; J_y = 0.05; J_z = 0.09; m = 0.8; g = 9.8; ell = 0.33; k_1 = 4; k_2 =0.05;
parameters = [J_x J_y J_z m g ell k_1 k_2];

% Seed with exact force to hover
initial_guess = m*g/k_1/4*ones(size(time_span,2),n_inputs); 

% Specify max number of iterations
n_iterations = 100;

% Construct ilqr object
ilqr_ = ilqr(init_state,target_state,initial_guess,dt,start_time,end_time,@calc_f_disc,@calc_A_disc,@calc_B_disc,Q_k,R_k,Q_T,parameters,n_iterations);
% Solve
[states,inputs,k_feedforward,K_feedback,final_cost] = ilqr_.solve();

save('quadcopter-trajectory.mat','states','inputs','dt','parameters','K_feedback','target_state');

figure(1);
plot3(states(:,1),states(:,2),states(:,3));
hold on
plot3(target_state(1),target_state(2),target_state(3),'ro');
legend('Optimal trajectory','Target state');

figure(2);
h1 = plot(time_span,inputs(:,1));
hold on
h2 = plot(time_span,inputs(:,2));
h3 = plot(time_span,inputs(:,3));
h4 = plot(time_span,inputs(:,4));
legend([h1,h2,h3,h4],'Front left','Front right','Back left','Back right');


