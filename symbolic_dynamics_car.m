clear all; close all;
syms px py v theta v_dot theta_dot dt

% Define the states and inputs
states = [px;py;v;theta];
inputs = [v_dot;theta_dot];

% Store any physical parameters needed here e.g. mass, lengths...
parameters = sym([]); 

% Define the dynamics
f = [v*cos(theta);
    v*sin(theta);
    v_dot;
    theta_dot];

% Eueler integration to define discrete update
f_disc = states + f*dt;

% Linearized discrete dynamical matrices
A_disc = jacobian(f_disc,states);
B_disc = jacobian(f_disc,inputs);


%% Write functions for the dynamics
matlabFunction(A_disc,'File','calc_A_disc','Vars',[{states},{inputs},{dt},{parameters}])
matlabFunction(B_disc,'File','calc_B_disc','Vars',[{states},{inputs},{dt},{parameters}])

matlabFunction(f,'File','calc_f','Vars',[{states},{inputs},{parameters}])
matlabFunction(f_disc,'File','calc_f_disc','Vars',[{states},{inputs},{dt},{parameters}])