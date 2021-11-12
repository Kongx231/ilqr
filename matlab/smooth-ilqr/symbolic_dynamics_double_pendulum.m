clear all; close all;
% Declare symbolic variables
syms q1 q2 q1_dot q2_dot real
syms tau_1 tau_2 real
syms m1 L1 m2 L2 g dt real

% Group syms into vectors
q = [q1;q2];
q_dot = [q1_dot;q2_dot];
% tau = [tau_1;tau_2];
tau = [tau_1;tau_2];
states = [q;q_dot];

inputs = tau;
% Compute generalized inertia matrices for each link
M1 = [m1 0 0;
    0 m1 0;
    0 0 1/12*m1*L1^2];
M2 = [m2 0 0;
    0 m2 0;
    0 0 1/12*m2*L2^2];

% Compute body Jacobians for each link
Jb_sL1 = [0 0;
    L1/2 0
    1 0];
Jb_sL2 = [L1*sin(q2) 0
    L1*cos(q2)+L2/2 L2/2
    1 1];

% Compute manipulator inertia tensor
M = simplify(Jb_sL1'*M1*Jb_sL1 + Jb_sL2'*M2*Jb_sL2);

% % Compute Coriolis matrix
C  = sym(zeros(length(q),length(q)));
for ii = 1:length(q)
    for jj = 1:length(q)
        for kk = 1:length(q)
            C(ii,jj) = C(ii,jj) + 1/2*(diff(M(ii,jj),q(kk)) + diff(M(ii,kk),q(jj)) - diff(M(jj,kk),q(ii)))*q_dot(kk);
        end
    end
end
C = simplify(C);

% Compute nonlinear and applied force terms (no constraints)
V = simplify(m1*g*L1/2*sin(q1) + m2*g*(L1*sin(q1) + L2/2*sin(q1+q2)));
N = jacobian(V, q)';
Y = tau;

f = [q_dot;M\(Y-N-C*q_dot)];

% Discretize the dynamics using euler integration
f_disc = states+f*dt;
% Take the jacobian with respect to states and inputs
A_disc = jacobian(f_disc,states);
B_disc = jacobian(f_disc,inputs);

% Define the parameters of the system
parameters = [m1 L1 m2 L2 g]; % mass of the link 1, length of link 1, mass of the link 2, length of link 2, acceleration due to gravity

matlabFunction(f_disc,'File','calc_f_disc','Vars',[{states},{inputs},{dt},{parameters}])
matlabFunction(A_disc,'File','calc_A_disc','Vars',[{states},{inputs},{dt},{parameters}])
matlabFunction(B_disc,'File','calc_B_disc','Vars',[{states},{inputs},{dt},{parameters}])


