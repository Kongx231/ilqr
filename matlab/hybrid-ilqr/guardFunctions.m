function [constraintFcns,isterminal,direction] = guardFunctions(t,x,u,calc_g,parameters)
% Compute constraint function (transition if goes negative)
a = calc_g(x,u,parameters);

constraintFcns = a; % The value that we want to be zero
isterminal = ones(length(constraintFcns), 1);  % Halt integration 
direction = 0*ones(length(a),1);   % The zero can be approached from either direction
end