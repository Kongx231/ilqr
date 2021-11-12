function salt1 = calc_salt12(in1,uy,in3)
%CALC_SALT12
%    SALT1 = CALC_SALT12(IN1,UY,IN3)

%    This function was generated by the Symbolic Math Toolbox version 8.6.
%    12-Nov-2021 01:22:01

e = in3(:,3);
g = in3(:,2);
m = in3(:,1);
y_dot = in1(2,:);
t2 = -e;
salt1 = reshape([t2,((uy-g.*m).*(e+1.0))./(m.*y_dot),0.0,t2],[2,2]);
