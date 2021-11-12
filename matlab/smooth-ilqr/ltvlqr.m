function K = ltvlqr(states,inputs,A,B,Q_k,R_k,Q_T,dt,parameters)
% This function (linear time varying linear quadratic regularizer) schedules feedback gains by using the discrete algebraic riccati equation 

% Get the number of timesteps, dimension of states and dimension of inputs
n_timesteps = size(inputs,1);
n_states = size(states,2);
n_inputs = size(inputs,2);

K = zeros(n_timesteps,n_inputs,n_states);
% Boundary condition
P_k = Q_T;
for ii=flip(1:n_timesteps) % Run backwards in time
    % Get the current state and input
    current_state = states(ii,:)';
    current_input = inputs(ii,:)';
    
    % Get the linearization about the state and input pair
    A_k = A(current_state,current_input,dt,parameters);
    B_k = B(current_state,current_input,dt,parameters);
    % Calculate the feedback gain and store it
    K_k = -inv(R_k+B_k'*P_k*B_k)*B_k'*P_k*A_k;
    K(ii,:,:) = K_k;
    % update for P
    P_k = Q_k+A_k'*P_k*A_k-A_k'*P_k*B_k*inv(R_k+B_k'*P_k*B_k)*B_k'*P_k*A_k;
end
end