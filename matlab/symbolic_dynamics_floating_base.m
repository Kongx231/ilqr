close all; clear all;
% states
syms x y z x_dot y_dot z_dot phi theta psi phi_dot_world theta_dot_world psi_dot_world dt real
syms foot_pos_1x foot_pos_1y foot_pos_1z real
syms foot_pos_2x foot_pos_2y foot_pos_2z real
syms foot_pos_3x foot_pos_3y foot_pos_3z real
syms foot_pos_4x foot_pos_4y foot_pos_4z real
% Inputs
syms F_1x F_1y F_1z real
syms F_2x F_2y F_2z real
syms F_3x F_3y F_3z real
syms F_4x F_4y F_4z real

% Parameters
syms I_xx I_yy I_zz m g real

% input mapping 
robot_pos = [x;y;z];
robot_orientation = [phi;theta;psi];
robot_lin_vel = [x_dot;y_dot;z_dot];
robot_ang_vel_world = [phi_dot_world;theta_dot_world;psi_dot_world];
states = [robot_pos;robot_orientation;robot_lin_vel;robot_ang_vel_world];

% Define the position of the feet
foot_pos_1 = [foot_pos_1x foot_pos_1y foot_pos_1z]';
foot_pos_2 = [foot_pos_2x foot_pos_2y foot_pos_2z]';
foot_pos_3 = [foot_pos_3x foot_pos_3y foot_pos_3z]';
foot_pos_4 = [foot_pos_4x foot_pos_4y foot_pos_4z]';

% Get the difference between the robot position and feet position
r_1 = robot_pos - foot_pos_1;
r_2 = robot_pos - foot_pos_2;
r_3 = robot_pos - foot_pos_3;
r_4 = robot_pos - foot_pos_4;

% Define the forces
F_1 = [F_1x F_1y F_1z]';
F_2 = [F_2x F_2y F_2z]';
F_3 = [F_3x F_3y F_3z]';
F_4 = [F_4x F_4y F_4z]';

inputs = [F_1;F_2;F_3;F_4];

% Create rpy rotation matrix
rot_roll = [1,0,0;
            0,cos(phi),-sin(phi);
            0,sin(phi),cos(phi)];
rot_pitch = [cos(theta),0,sin(theta);
            0,1,0;
            -sin(theta),0,cos(theta)];
rot_yaw = [cos(psi),-sin(psi),0;
            sin(psi),cos(psi),0;
            0, 0, 1];
rot_mat = rot_yaw*rot_pitch*rot_roll;

% Define the body inertia and then map it to world inertia (eq 14)
I = diag([I_xx,I_yy,I_zz]);
I_hat = rot_mat'*I*rot_mat;
I_hat_inv = simplify(inv(I_hat));

% Map body to world angular velocities (eq 10)
rot_mat_vel = [cos(theta)*cos(psi),-sin(psi),0;
                cos(theta)*sin(psi),cos(psi),0;
                0,0,1];
            
ang_vel_body = simplify(inv(rot_mat_vel)*robot_ang_vel_world);
% Get angular acceleration from the wrenches (we use the same approximation
% as eq 13) (eq 6)
ang_accel_world = simplify(I_hat_inv*(skew(r_1)*F_1 + skew(r_2)*F_2 + skew(r_3)*F_3 + skew(r_4)*F_4));

% Get linear accleartion from the forces (eq 5)
gravity_vec = [0;0;g];

linear_accel = 1/m*(F_1+F_2+F_3+F_4)+gravity_vec;

% Ordering of states, position, ang position, linear vel, angular vel
f = [robot_lin_vel;ang_vel_body;linear_accel;ang_accel_world];

% Eueler integration to define discrete update
f_disc = states + f*dt;

% Linearized discrete dynamical matrices
A_disc = jacobian(f_disc,states);
B_disc = jacobian(f_disc,inputs);

% Parameters
parameters = [I_xx I_yy I_zz m g...
              foot_pos_1x foot_pos_1y foot_pos_1z...
              foot_pos_2x foot_pos_2y foot_pos_2z...
              foot_pos_3x foot_pos_3y foot_pos_3z...
              foot_pos_4x foot_pos_4y foot_pos_4z];

%% Write functions for the dynamics
matlabFunction(A_disc,'File','calc_A_disc','Vars',[{states},{inputs},{dt},{parameters}])
matlabFunction(B_disc,'File','calc_B_disc','Vars',[{states},{inputs},{dt},{parameters}])

matlabFunction(f,'File','calc_f','Vars',[{states},{inputs},{parameters}])
matlabFunction(f_disc,'File','calc_f_disc','Vars',[{states},{inputs},{dt},{parameters}])