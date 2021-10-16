close all; clear all;
syms mc mp ll g x x_dot theta theta_dot u dt real

% Define the states and inputs
inputs = u;
states = [x;theta;x_dot;theta_dot];

% states x,theta,x_dot,theta_dot
f = [x_dot;
    theta_dot;
    1/(mc+mp*sin(theta)^2)*(u+mp*sin(theta)*(ll*theta_dot^2+g*cos(theta)));
    1/(ll*(mc+mp*sin(theta)^2))*(-u*cos(theta)-mp*ll*theta_dot^2*cos(theta)*sin(theta)-(mc+mp)*g*sin(theta))];


% Discretize the dynamics using euler integration
f_disc = states+f*dt;
% Take the jacobian with respect to states and inputs
A_disc = jacobian(f_disc,states);
B_disc = jacobian(f_disc,inputs);

% Define the parameters of the system
parameters = [mc,mp,ll,g]; % mass of the cart, mass of the pole, length of the pole, acceleration due to gravity

matlabFunction(f_disc,'File','calc_f_disc','Vars',[{states},{inputs},{dt},{parameters}])
matlabFunction(A_disc,'File','calc_A_disc','Vars',[{states},{inputs},{dt},{parameters}])
matlabFunction(B_disc,'File','calc_B_disc','Vars',[{states},{inputs},{dt},{parameters}])