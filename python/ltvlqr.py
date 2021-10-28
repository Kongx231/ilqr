import numpy as np

def ltvlqr(states,inputs,A,B,Q_k,R_k,Q_T,dt,parameters):
    # This function (linear time varying linear quadratic regularizer)
    # schedules feedback gains by using the discrete algebraic riccati equation
    
    # Get the number of timesteps, dimension of states and dimension of inputs
    n_timesteps = np.shape(inputs)[0]
    n_states = np.shape(states)[1]
    n_inputs = np.shape(inputs)[1]
    
    K_feedback = np.zeros((n_timesteps,n_inputs,n_states))
    # Boundary condition
    P_k = Q_T
    for ii in reversed(range(0, n_timesteps)):# Run backwards in time
        # Get the current state and input
        current_state = states[ii,:].flatten()
        current_input = inputs[ii,:].flatten()
        
        # Get the linearization about the state and input pair
        A_k = A(current_state,current_input,dt,parameters)
        B_k = B(current_state,current_input,dt,parameters)
        # Calculate the feedback gain and store it
        K_k = -np.linalg.inv(R_k+B_k.T@P_k@B_k)@B_k.T@P_k@A_k
        K_feedback[ii,:,:] = K_k
        # update for P
        P_k = Q_k+A_k.T@P_k@A_k-A_k.T@P_k@B_k@np.linalg.inv(R_k+B_k.T@P_k@B_k)@B_k.T@P_k@A_k
    return K_feedback