close all; clear all;
% states
syms px py pz u v w phi theta psi p q r dt real
% Inputs
% syms F_l F_r F_b F_f tau_phi tau_theta tau_psi
% syms delta_l delta_r delta_b delta_f k_1 k_2 real
syms delta_l delta_r delta_b delta_f k_1 k_2 real
% Parameters
syms J_x J_y J_z m g ell k_1 k_2 real

% input mapping 
F_l = k_1*delta_l; F_r = k_1*delta_r; F_b = k_1*delta_b; F_f = k_1*delta_f;
tau_l = k_2*delta_l; tau_r = k_2*delta_r; tau_b = k_2*delta_b; tau_f = k_2*delta_f;

F = F_f+F_r +F_b+F_l;
tau_phi = ell*(F_l-F_r);
tau_theta = ell*(F_f-F_b);
tau_psi = -tau_f+tau_r -tau_b+tau_l;

trans_vel = [cos(theta)*cos(psi) ,  sin(phi)*sin(theta)*cos(psi) - cos(phi)*sin(psi) , cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi);
    cos(theta)*sin(psi) ,  sin(phi)*sin(theta)*sin(psi) + cos(phi)*cos(psi) , cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi);
    sin(theta),-sin(phi)*cos(theta) , -cos(phi)*cos(theta)]*[u; v; w];
trans_vel(3) = -trans_vel(3);
trans_accel = [r*v - q*w ; p*w - r*u ; q*u - p*v] +...
    [-g*sin(theta) ; g*cos(theta)*sin(phi) ; g*cos(theta)*cos(phi)] +...
    (1/m)*[0 ; 0 ; -F];
angular_vel = [1 , sin(phi)*tan(theta) , cos(phi)*tan(theta) ;
    0 , cos(phi) , -sin(phi) ;
    0 , sin(phi)/cos(theta) , cos(phi)/cos(theta)]* [p;q;r];
angular_accel= [(J_y-J_z)/J_x*q*r;
    (J_z-J_x)/J_y*p*r;
    (J_x-J_y)/J_z*p*q] + ...
    [1/J_x*tau_phi;
    1/J_y*tau_theta;
    1/J_z*tau_psi];

states = [px py pz phi theta psi u v w p q r]';
% inputs = [F tau_phi tau_theta tau_psi]';
inputs = [delta_f delta_r delta_b delta_l]';

f = [trans_vel;angular_vel;trans_accel;angular_accel];

% Eueler integration to define discrete update
f_disc = states + f*dt;

% Linearized discrete dynamical matrices
A_disc = jacobian(f_disc,states);
B_disc = jacobian(f_disc,inputs);

% Parameters
parameters = [J_x J_y J_z m g ell k_1 k_2];

%% Write functions for the dynamics
matlabFunction(A_disc,'File','calc_A_disc','Vars',[{states},{inputs},{dt},{parameters}])
matlabFunction(B_disc,'File','calc_B_disc','Vars',[{states},{inputs},{dt},{parameters}])

matlabFunction(f,'File','calc_f','Vars',[{states},{inputs},{parameters}])
matlabFunction(f_disc,'File','calc_f_disc','Vars',[{states},{inputs},{dt},{parameters}])