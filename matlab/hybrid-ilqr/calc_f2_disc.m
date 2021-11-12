function f2_disc = calc_f2_disc(in1,uy,dt,in4)
%CALC_F2_DISC
%    F2_DISC = CALC_F2_DISC(IN1,UY,DT,IN4)

%    This function was generated by the Symbolic Math Toolbox version 8.6.
%    12-Nov-2021 01:22:01

g = in4(:,2);
m = in4(:,1);
y = in1(1,:);
y_dot = in1(2,:);
f2_disc = [y+dt.*y_dot;y_dot+(dt.*(uy-g.*m))./m];