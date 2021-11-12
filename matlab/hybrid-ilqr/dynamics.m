function dx = dynamics(t,x,f,input,parameters)
% Initialize time derivative of state vector as column vector
    dx = f(x,input,parameters);
end