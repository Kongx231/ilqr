function [constraintFcns,isterminal,direction] = guardFunctions(t,x,calc_g,parameters)
% Compute constraint function (transition if goes negative)
a = calc_g(x);

constraintFcns = a; % The value that we want to be zero
isterminal = ones(length(constraintFcns), 1);  % Halt integration 
direction = -ones(length(a),1);   % The zero can be approached from either direction
end