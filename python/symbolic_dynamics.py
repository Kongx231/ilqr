import numpy as np
import sympy as sp
from sympy.matrices import Matrix

def symbolic_dynamics_pendulum():
    m,g,L,theta,theta_dot,u,dt = sp.symbols('m g L theta theta_dot u dt')

    # Define the states and inputs
    inputs = Matrix([u])
    states = Matrix([theta,theta_dot])
    # Defining the dynamics of the system
    f = Matrix([theta_dot,(u-m*g*L*sp.sp.sin(theta))/(m*L*L)])

    # Discretize the dynamics usp.sing euler integration
    f_disc = states+f*dt
    # Take the jacobian with respect to states and inputs
    A_disc = f_disc.jacobian(states)
    B_disc = f_disc.jacobian(inputs)

    # Define the parameters of the system
    parameters = Matrix([m,g,L])

    f_disc_func = sp.lambdify((states,inputs,dt,parameters),f_disc)
    A_disc_func = sp.lambdify((states,inputs,dt,parameters),A_disc)
    B_disc_func = sp.lambdify((states,inputs,dt,parameters),B_disc)
    return (f_disc_func,A_disc_func,B_disc_func)

def symbolic_dynamics_acrobot():
    print("Importing symbolic dynamics")
    # Declare symbolic variables
    q1,q2,q1_dot,q2_dot = sp.symbols('q1 q2 q1_dot q2_dot')
    tau_2 = sp.symbols('tau_2')
    m1,L1,m2,L2,g,dt = sp.symbols('m1 L1 m2 L2 g dt ')
    
    # Group syms into vectors
    q = Matrix([[q1],[q2]])
    q_dot = Matrix([[q1_dot],[q2_dot]])
    # tau = [tau_1tau_2]
    tau = Matrix([[0],[tau_2]])
    states = Matrix([[q],[q_dot]])
    
    inputs = Matrix([tau_2])
    # Compute generalized inertia matrices for each link
    M1 = Matrix([[m1,0,0],
        [0,m1,0],
        [0,0,1/12*m1*L1*L1]])
    M2 = Matrix([[m2,0,0],
        [0,m2,0],
        [0,0,1/12*m2*L2*L2]])
    
    # Compute body Jacobians for each link
    Jb_sL1 = Matrix([[0,0],
        [L1/2,0],
        [1,0]])
    Jb_sL2 = Matrix([[L1*sp.sin(q2),0],
        [L1*sp.cos(q2)+L2/2,L2/2],
        [1,1]])
    
    # Compute manipulator inertia tensor
    M = sp.simplify(Jb_sL1.T@M1@Jb_sL1 + Jb_sL2.T@M2@Jb_sL2)
    
    # # Compute Coriolis matrix
    n_config = 2
    C  = sp.zeros(n_config,n_config)
    for ii in range(0,n_config):
        for jj in range(0,n_config):
            for kk in range(0,n_config):
                C[ii,jj] = C[ii,jj] + 1/2*(sp.diff(M[ii,jj],q[kk]) + sp.diff(M[ii,kk],q[jj]) - sp.diff(M[jj,kk],q[ii]))*q_dot[kk]


    
    # Compute nonlinear and applied force terms (no constraints)
    V = Matrix([(m1*g*L1/2*sp.sin(q1) + m2*g*(L1*sp.sin(q1) + L2/2*sp.sin(q1+q2)))])
    N = V.jacobian(q).T
    Y = tau

    f = Matrix([q_dot,M.inv()@(Y-N-C@q_dot)])

    # Discretize the dynamics usp.sing euler integration
    f_disc= states+f*dt
    # Take the jacobian with respect to states and inputs
    A_disc = f_disc.jacobian(states)
    B_disc = f_disc.jacobian(inputs)
    
    # Define the parameters of the system
    parameters = Matrix([m1,L1,m2,L2,g]) # mass of the link 1, length of link 1, mass of the link 2, length of link 2, acceleration due to gravity
    
    f_disc_func = sp.lambdify((states,inputs,dt,parameters),f_disc)
    A_disc_func = sp.lambdify((states,inputs,dt,parameters),A_disc)
    B_disc_func = sp.lambdify((states,inputs,dt,parameters),B_disc)
    print("Symbolic dynamics imported")
    return (f_disc_func,A_disc_func,B_disc_func)





