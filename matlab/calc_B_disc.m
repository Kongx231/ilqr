function B_disc = calc_B_disc(in1,tau_2,dt,in4)
%CALC_B_DISC
%    B_DISC = CALC_B_DISC(IN1,TAU_2,DT,IN4)

%    This function was generated by the Symbolic Math Toolbox version 8.6.
%    04-Nov-2021 01:24:17

L1 = in4(:,2);
L2 = in4(:,4);
m1 = in4(:,1);
m2 = in4(:,3);
q2 = in1(2,:);
t2 = cos(q2);
t3 = L1.^2;
t4 = t2.^2;
t5 = m1.*t3.*4.0;
t6 = m2.*t3.*1.2e+1;
t7 = m2.*t3.*t4.*9.0;
t8 = -t7;
t9 = t5+t6+t8;
t10 = 1.0./t9;
B_disc = [0.0;0.0;(dt.*t10.*(L2.*4.0+L1.*t2.*6.0).*-3.0)./L2;(1.0./L2.^2.*dt.*t10.*(t5+t6+L2.^2.*m2.*4.0+L1.*L2.*m2.*t2.*1.2e+1).*3.0)./m2];
