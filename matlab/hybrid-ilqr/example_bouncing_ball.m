close all; clear all;
% Compute symbolic dynamics and create functions
% symbolic_dynamics_1d_bouncing_ball();

% Initialize timings
dt = 0.001;
start_time = 0;
end_time = 1;
time_span = start_time:dt:end_time;

% Set desired state
n_states = 2;
n_inputs = 1;
init_state = [4;0]; % Define the initial state to be the origin with no velocity
init_mode = 1;
target_state = [3;0]; % Swing pendulum upright

% Initial guess of zeros, but you can change it to any guess
initial_guess = 0.0*ones(size(time_span,2),1);
% Define weighting matrices
Q_k = zeros(n_states,n_states); % zero weight to penalties along a strajectory since we are finding a trajectory
R_k = 0.00025*eye(n_inputs);

% Set the terminal cost
Q_T = 40*eye(n_states);

% Set the physical parameters of the system
mass = 1;
gravity = 9.8;
coefficient_restitution = 0.75;
parameters = [mass,gravity,coefficient_restitution];

% Specify max number of iterations
n_iterations = 50;

% Specify optimality condition
min_reduction = 1e-2;

% Define optimization struct
optimization_struct.init_state = init_state;
optimization_struct.init_mode = init_mode;
optimization_struct.target_state = target_state;
optimization_struct.initial_guess = initial_guess;
optimization_struct.dt = dt;
optimization_struct.start_time = start_time;
optimization_struct.end_time = end_time;
optimization_struct.n_iterations = n_iterations;
optimization_struct.min_reduction = min_reduction;
optimization_struct.Q_T = Q_T;
optimization_struct.Q_k = Q_k;
optimization_struct.R_k = R_k;

% Construct ilqr object
A_disc = {@calc_A_lin1_disc,@calc_A_lin2_disc};
B_disc = {@calc_B_lin1_disc,@calc_B_lin2_disc};
f = {@calc_f1,@calc_f2};
resets = {@calc_r12,@calc_r21};
guards = {@calc_g12,@calc_g21};
salts = {@calc_salt12,@calc_salt21};

% Define dynamics struct
dynamics_struct.A_disc = A_disc;
dynamics_struct.B_disc = B_disc;
dynamics_struct.f = f;
dynamics_struct.resets = resets;
dynamics_struct.guards = guards;
dynamics_struct.salts = salts;
dynamics_struct.parameters = parameters;

ilqr_ = h_ilqr(optimization_struct,dynamics_struct);
% Solve
[states,inputs,k_feedforward,K_feedback,final_cost,expected_reduction] = ilqr_.solve();
% save('swingup-trajectory.mat','states','inputs','dt','parameters','k_feedforward','K_feedback'); % Save trajectory to track later

figure(1);
h1 = plot(states(:,1),states(:,2),'k');
hold on
h2 = plot(target_state(1),target_state(2),'ro');
legend([h1,h2],"Final Trajectory","Target State");
xlabel('$$\theta$$');
ylabel('$$\dot{\theta}$$');

figure(2);
h1 = plot(inputs(:,1),'k');
xlabel('Timestep');
ylabel('$$u$$');

figure(3);
animate_bouncing_ball(states,dt,inputs)

