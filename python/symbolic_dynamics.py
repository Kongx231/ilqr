import numpy as np
import sympy as sp
from sympy.matrices import Matrix

def symbolic_dynamics_pendulum():
    m_sym,g_sym,L_sym,theta_sym,theta_dot_sym,u_sym,dt_sym = sp.symbols('m g L theta theta_dot u dt')

    # Define the states and inputs
    inputs_sym = Matrix([u_sym])
    states_sym = Matrix([theta_sym,theta_dot_sym])
    # Defining the dynamics of the system
    f_sym = Matrix([theta_dot_sym,(u_sym-m_sym*g_sym*L_sym*sp.sin(theta_sym))/(m_sym*L_sym*L_sym)])

    # Discretize the dynamics using euler integration
    f_disc_sym = states_sym+f_sym*dt_sym
    # Take the jacobian with respect to states and inputs
    A_disc_sym = f_disc_sym.jacobian(states_sym)
    B_disc_sym = f_disc_sym.jacobian(inputs_sym)

    # Define the parameters of the system
    parameters_sym = Matrix([m_sym,g_sym,L_sym])

    f_disc_func = sp.lambdify((states_sym,inputs_sym,dt_sym,parameters_sym),f_disc_sym)
    A_disc_func = sp.lambdify((states_sym,inputs_sym,dt_sym,parameters_sym),A_disc_sym)
    B_disc_func = sp.lambdify((states_sym,inputs_sym,dt_sym,parameters_sym),B_disc_sym)
    return (f_disc_func,A_disc_func,B_disc_func)


