clear all; close all;
syms m dadx dady g y y_dot uy dt coefficients e real 
% Define the configuration and velocity variables
q_dot = y_dot;
q = y;
states = [q;q_dot];

% Define the ground to be at 0 height
a = -y;
% Jacobian of the ground constraint
A = jacobian(a,y);
% Defining the mass matrix
M = m*eye(1);
% Defining the block matrix when solving for constrained dynamics and reset
block_mat = [M,A';
            A,0];
block_mat_inv = inv(block_mat);

% Gravity
N = m*g;
% Thruster input
Y = uy;
inputs = Y;

q_ddot = inv(M)*(Y-N);

% Defining the dynamics of the system, since bouncing ball it is always
% flight
f1 = [q_dot;q_ddot];
f2 = [q_dot;q_ddot];
% Defining the linlearized dynamics of the system

A_lin1 = jacobian(f1,states);
B_lin1 = jacobian(f1,inputs);
A_lin2 = jacobian(f2,states);
B_lin2 = jacobian(f2,inputs);

%%
% Calculating reset maps
impact_vars = block_mat_inv*[M;-e*A]*q_dot; % Stores the post impact velocity and impulse
R1 = [y;impact_vars(1)];
R2 = states; % identity change

DR1 = jacobian(R1,states);
DR2 = jacobian(R2,states);
% Guards
G1 = a;
DG1 = jacobian(G1,states);
G2 = -y_dot;
DG2 = jacobian(G2,states);

% No time derivatives for the guards
DtG1 = sym(0);
DtG2 = sym(0);

parameters = [m,g,e];
%%
% New matlabfunction calls just to be consistent
matlabFunction(f2,'File','calc_f2','Vars',[{states},{inputs},{parameters}])
matlabFunction(f1,'File','calc_f1','Vars',[{states},{inputs},{parameters}])
matlabFunction(R1,'File','calc_r12','Vars',[{states},{inputs},{parameters}])
matlabFunction(DR1,'File','calc_Dr12','Vars',[{states},{inputs},{parameters}])
matlabFunction(R2,'File','calc_r21','Vars',[{states},{inputs},{parameters}])
matlabFunction(DR2,'File','calc_Dr21','Vars',[{states},{inputs},{parameters}])
matlabFunction(G1,'File','calc_g12','Vars',[{states},{inputs},{parameters}])
matlabFunction(DG1,'File','calc_Dg12','Vars',[{states},{inputs},{parameters}])
matlabFunction(DtG1,'File','calc_Dtg12','Vars',[{states},{inputs},{parameters}])
matlabFunction(G2,'File','calc_g21','Vars',[{states},{inputs},{parameters}])
matlabFunction(DG2,'File','calc_Dg21','Vars',[{states},{inputs},{parameters}])
matlabFunction(DtG2,'File','calc_Dtg21','Vars',[{states},{inputs},{parameters}])

matlabFunction(G1,'File','calc_a1','Vars',[{states},{inputs},{parameters}])
matlabFunction(G2,'File','calc_a2','Vars',[{states},{inputs},{parameters}])
% matlabFunction(Dg2,'File','calc_Dg21')
matlabFunction(sym(A_lin1),'File','calc_A_lin1','Vars',[{states},{inputs},{parameters}])
matlabFunction(sym(A_lin2),'File','calc_A_lin2','Vars',[{states},{inputs},{parameters}])
matlabFunction(sym(B_lin1),'File','calc_B_lin1','Vars',[{states},{inputs},{parameters}])
matlabFunction(sym(B_lin2),'File','calc_B_lin2','Vars',[{states},{inputs},{parameters}])
% % matlabFunction(f,'File','calc_f2','Vars',[{states},{parameters}])
% matlabFunction(f1,'File','calc_f1','Vars',[{states},{uy},{parameters}])
% matlabFunction(reset,'File','calc_r12','Vars',[{states},{parameters}])
% matlabFunction(DR,'File','calc_Dr12','Vars',{parameters})
% matlabFunction(a,'File','calc_g12','Vars',[{states}])
% matlabFunction(Dg,'File','calc_Dg12')
% matlabFunction(coefficients,'File','calc_constraint_coeff')
% matlabFunction(a,'File','calc_a1','Vars',{states})
% matlabFunction(y_dot,'File','calc_a2','Vars',{states})
% 
% matlabFunction(A_lin1,'File','calc_Df1')
% matlabFunction(B_lin1,'File','calc_B1','Vars',{parameters})
%% Discrete
f1_disc = states+f1*dt;
f2_disc = states+f2*dt;
A_lin2_disc = jacobian(f2_disc,states);
A_lin1_disc = jacobian(f1_disc,states);
B_lin2_disc = jacobian(f2_disc,inputs);
B_lin1_disc = jacobian(f1_disc,inputs);

matlabFunction(f2_disc,'File','calc_f2_disc','Vars',[{states},{inputs},{dt},{parameters}])
matlabFunction(f1_disc,'File','calc_f1_disc','Vars',[{states},{inputs},{dt},{parameters}])
matlabFunction(A_lin1_disc,'File','calc_A_lin1_disc','Vars',[{states},{inputs},{dt},{parameters}])
matlabFunction(A_lin2_disc,'File','calc_A_lin2_disc','Vars',[{states},{inputs},{dt},{parameters}])
matlabFunction(B_lin1_disc,'File','calc_B_lin1_disc','Vars',[{states},{inputs},{dt},{parameters}])
matlabFunction(B_lin2_disc,'File','calc_B_lin2_disc','Vars',[{states},{inputs},{dt},{parameters}])

% Saltation matrix calculation
salt1 = simplify(DR1 + (calc_f2(R1,inputs,parameters)-DR1*f1)*DG1/(DtG1+DG1*f1));
salt2 = DR2; % Not bother calculating becuase it is identity
matlabFunction(salt1,'File','calc_salt12','Vars',[{states},{inputs},{parameters}])
matlabFunction(salt2,'File','calc_salt21','Vars',[{states},{inputs},{parameters}])