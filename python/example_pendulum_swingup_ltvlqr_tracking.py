import numpy as np

# Import iLQR class
from ilqr import ilqr
# Import ltvlqr function
from ltvlqr import ltvlqr
# Import pendulum dynamics
from symbolic_dynamics import symbolic_dynamics_pendulum
# Import animator
from animate import animate_pendulum, animate_pendulum_tracking

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
perturbed_init_state = init_state + np.array([-np.pi/4,0])
current_state = perturbed_init_state
perturbed_states[0,:] = current_state.copy()
# Simulate
for ii in range(0,ilqr_.n_timesteps_):
    # Compute error and feedback
    current_error = (current_state - states[ii,:]).flatten()
    feedback = new_K[ii,:,:]@current_error
    current_input = inputs[ii,:] + feedback
    next_state = f(current_state,current_input,dt,parameters).flatten()

    perturbed_states[ii + 1, :] = next_state.copy()
    perturbed_inputs[ii,:] = current_input.copy()
    # Update the state
    current_state = next_state.copy()

animate_pendulum_tracking(states,perturbed_states,inputs,dt,parameters)
