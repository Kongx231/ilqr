clear all; close all;
syms m g L theta theta_dot u dt real 

% Define the states and inputs
inputs = u;
states = [theta;theta_dot];
% Defining the dynamics of the system
f = [theta_dot;(u-m*g*L*sin(theta))/(m*L^2)];

% Discretize the dynamics using euler integration
f_disc = states+f*dt;
% Take the jacobian with respect to states and inputs
A_disc = jacobian(f_disc,states);
B_disc = jacobian(f_disc,inputs);

% Define the parameters of the system
parameters = [m,g,L];

matlabFunction(f_disc,'File','calc_f_disc','Vars',[{states},{inputs},{dt},{parameters}])
matlabFunction(A_disc,'File','calc_A_disc','Vars',[{states},{inputs},{dt},{parameters}])
matlabFunction(B_disc,'File','calc_B_disc','Vars',[{states},{inputs},{dt},{parameters}])

