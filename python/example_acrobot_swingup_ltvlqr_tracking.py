import numpy as np

# Import iLQR class
from ltvlqr import ltvlqr
# Import pendulum dynamics
from symbolic_dynamics import symbolic_dynamics_acrobot
# Import animator
from animate import animate_acrobot, animate_acrobot_tracking

# Import dynamics
(f,A,B) = symbolic_dynamics_acrobot()

# Initialize timings
dt = 0.001
start_time = 0
end_time = 4
time_span = np.arange(start_time, end_time, dt).flatten()

# Set desired state
n_states = 4
n_inputs = 1
init_state = np.array([-np.pi/2,0,0,0])    # Define the initial state to be pointing down
target_state = np.array([np.pi/2,0,0,0])  # Swing acrobot upright

# Load trajectory for swing up
states = np.loadtxt('acrobot_swingup_states.csv',dtype=float,delimiter=',')
inputs = np.loadtxt('acrobot_swingup_inputs.csv',dtype=float,delimiter=',')
inputs = inputs.reshape((np.shape(inputs)[0],1))
# Parameters of the cart pole
mass1 = 1
length1 = 1
mass2 = 2 # more mass to swing better
length2 = 1
gravity = 9.8

parameters = np.array([mass1,length1,mass2,length2,gravity])
# Create a new gain schedule with linear time varying LQR with different weights (say we want better tracking)
# Define weighting matrices
Q_k = 0.1*np.eye(n_states) # zero weight to penalties along a strajectory since we are finding a trajectory
R_k = 0.01*np.eye(n_inputs)

# Set the terminal cost
Q_T = 100*np.eye(n_states)

new_K = ltvlqr(states,inputs,A,B,Q_k,R_k,Q_T,dt,parameters)

## Simulate with perturbation
# Initialze new input and state trajectories
perturbed_states = states.copy()
perturbed_inputs = inputs.copy()

# Add perturbation
perturbed_init_state = init_state + np.array([-np.pi/8,0,0,0])
current_state = perturbed_init_state
perturbed_states[0,:] = current_state.copy()
# Simulate
for ii in range(0,np.shape(inputs)[0]):
    # Compute error and feedback
    current_error = (current_state - states[ii,:]).flatten()
    feedback = new_K[ii,:,:]@current_error
    current_input = inputs[ii,:] + feedback
    next_state = f(current_state,current_input,dt,parameters).flatten()

    perturbed_states[ii + 1, :] = next_state.copy()
    perturbed_inputs[ii,:] = current_input.copy()
    # Update the state
    current_state = next_state.copy()
# Animate
animate_acrobot_tracking(states,perturbed_states,inputs,dt,parameters)





