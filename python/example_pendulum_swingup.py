import numpy as np

# Import iLQR class
from ilqr import ilqr
# Import pendulum dynamics
from symbolic_dynamics import symbolic_dynamics_pendulum
# Import animator
from animate import animate_pendulum

# Import dynamics
(f,A,B) = symbolic_dynamics_pendulum()

# Initialize timings
dt = 0.005
start_time = 0
end_time = 5
time_span = np.arange(start_time, end_time, dt).flatten()

# Set desired state
n_states = 2
n_inputs = 1
init_state = np.array([0,0])    # Define the initial state to be the origin with no velocity
target_state = np.array([np.pi,0])  # Swing pendulum upright

# Initial guess of zeros, but you can change it to any guess
initial_guess = 0.1*np.ones((np.shape(time_span)[0],n_inputs))
# Define weighting matrices
Q_k = np.zeros((n_states,n_states)) # zero weight to penalties along a strajectory since we are finding a trajectory
R_k = 0.001*np.eye(n_inputs)

# Set the terminal cost
Q_T = 100*np.eye(n_states)

# Set the physical parameters of the system
mass = 1
gravity = 9.8
pendulum_length = 1
parameters = np.array([mass,gravity,pendulum_length])

# Specify max number of iterations
n_iterations = 50

# Initialize ilqr object
ilqr_ = ilqr(init_state,target_state,initial_guess,dt,start_time,end_time,f,A,B,Q_k,R_k,Q_T,parameters,n_iterations)

# Solve for swing up
(states,inputs,k_feedforward,K_feedback,current_cost) = ilqr_.solve()

# Animate
animate_pendulum(states,inputs,dt,parameters)
