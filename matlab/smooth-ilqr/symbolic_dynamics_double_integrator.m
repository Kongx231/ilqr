clear all; close all;
% Declare symbolic variables
syms x x_dot real
syms force real
syms m dt real

states = [x;x_dot];
inputs = [force];
% Group syms into vectors
f = [x_dot;force/m];

% Discretize the dynamics using euler integration
f_disc = states+f*dt;
% Take the jacobian with respect to states and inputs
A_disc = jacobian(f_disc,states);
B_disc = jacobian(f_disc,inputs);

% Define the parameters of the system
parameters = [m]; % mass of the link 1, length of link 1, mass of the link 2, length of link 2, acceleration due to gravity

matlabFunction(f_disc,'File','calc_f_disc','Vars',[{states},{inputs},{dt},{parameters}]);
matlabFunction(A_disc,'File','calc_A_disc','Vars',[{states},{inputs},{dt},{parameters}]);
matlabFunction(B_disc,'File','calc_B_disc','Vars',[{states},{inputs},{dt},{parameters}]);


