clear all; close all;
syms m dadx dady g x y x_dot y_dot u_x u_y dt real 
q_dot = [x_dot;y_dot];
q = [x;y];
% Why do I even do this..........
states = [q;q_dot];
inputs = [u_x;u_y];

% Define the constact surface of the circle
a = (x).^2+(y-5).^2-2^2;

% Jacobian of the ground constraint
A = jacobian(a,q);
dA = sym(zeros(size(A)));
for i = 1:length(q)
    dA = dA + diff(A, q(i))*q_dot(i);
end

% Defining the mass matrix
M = m*eye(2);
% Defining the block matrix when solving for constrained dynamics and reset
block_mat = [M,A';
            A,0];
block_mat_inv = inv(block_mat);

% Gravity
N = [0;m*g];
% Thruster input
Y = [u_x;u_y];


q_ddot1 = inv(M)*(Y-N);

% Defining the dynamics of the system, since bouncing ball it is always
% flight
f1 = [q_dot;q_ddot1];

% Defining the contact dynamics
sol = block_mat_inv*([Y-N;zeros(size(A,1))]-[zeros(size(N,1),size(q_dot,1));dA]*q_dot);
q_ddot2 =sol(1:2);
lambda = simplify(sol(3));
lambda_x = jacobian(lambda,states);
lambda_u = jacobian(lambda,inputs);

f2 = [q_dot;q_ddot2];
% Defining the linlearized dynamics of the system

A_lin1 = jacobian(f1,states);
B_lin1 = jacobian(f1,inputs);
A_lin2 = jacobian(f2,states);
B_lin2 = jacobian(f2,inputs);

%%
% Calculating reset maps
impact_vars = simplify(block_mat_inv*[M;-0*A]*q_dot); % Stores the post impact velocity and impulse
R1 = [q;impact_vars(1:2)];
R2 = states; % identity change

DR1 = jacobian(R1,states);
DR2 = jacobian(R2,states);
% Guards
G1 = a;
DG1 = jacobian(G1,states);

% Liftoff is the guard for contact
G2 = lambda;
DG2 = jacobian(G2,states);

% No time derivatives for the guards
DtG1 = sym(0);
DtG2 = sym(0);

parameters = [m,g];
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