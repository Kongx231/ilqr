import numpy as np

# Import iLQR class
from ilqr import ilqr
# Import pendulum dynamics
from symbolic_dynamics import symbolic_dynamics_acrobot
# Import animator
from animate import animate_acrobot, animate_pendulum_tracking

# Import dynamics
(f,A,B) = symbolic_dynamics_acrobot()

# Initialize timings
dt = 0.001
start_time = 0
end_time = 3
time_span = np.arange(start_time, end_time, dt).flatten()

# Set desired state
n_states = 4
n_inputs = 1
init_state = np.array([-np.pi/2,0,0,0])    # Define the initial state to be pointing down
target_state = np.array([np.pi/2,0,0,0])  # Swing acrobot upright

# Initial random seed, but you can change it to any guess
np.random.seed(1)
initial_guess = 0.5*np.random.normal(loc = 0,scale = 1,size = (np.shape(time_span)[0],n_inputs))
# Define weighting matrices
Q_k = np.zeros((n_states,n_states)) # zero weight to penalties along a strajectory since we are finding a trajectory
R_k = 0.01*np.eye(n_inputs)

# Set the terminal cost
Q_T = 10*np.eye(n_states)
# We care about terminal positions the most
Q_T[0,0] = 1000
Q_T[1,1] = 1000

# Parameters of the cart pole
mass1 = 1
length1 = 1
mass2 = 2 # more mass to swing better
length2 = 1
gravity = 9.8

parameters = np.array([mass1,length1,mass2,length2,gravity])

# Specify max number of iterations
n_iterations = 50

# Initialize ilqr object
ilqr_ = ilqr(init_state,target_state,initial_guess,dt,start_time,end_time,f,A,B,Q_k,R_k,Q_T,parameters,n_iterations)

# Solve for swing up
(states,inputs,k_feedforward,K_feedback,current_cost) = ilqr_.solve()

np.savetxt('acrobot_swingup_states.csv',states,delimiter=',')
np.savetxt('acrobot_swingup_inputs.csv',inputs,delimiter=',')
# Animate
animate_acrobot(states,inputs,dt,parameters)

