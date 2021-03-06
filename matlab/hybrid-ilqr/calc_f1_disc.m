function f1_disc = calc_f1_disc(in1,in2,dt,in4)
%CALC_F1_DISC
%    F1_DISC = CALC_F1_DISC(IN1,IN2,DT,IN4)

%    This function was generated by the Symbolic Math Toolbox version 8.6.
%    19-Nov-2021 19:55:09

L1 = in4(:,2);
L2 = in4(:,4);
g = in4(:,5);
m1 = in4(:,1);
m2 = in4(:,3);
q1 = in1(1,:);
q2 = in1(2,:);
q1_dot = in1(3,:);
q2_dot = in1(4,:);
tau_1 = in2(1,:);
tau_2 = in2(2,:);
t2 = cos(q1);
t3 = cos(q2);
t4 = sin(q2);
t5 = q1+q2;
t6 = L1.^2;
t7 = L1.^3;
t8 = L2.^2;
t9 = L2.^3;
t10 = m2.^2;
t11 = q1_dot.^2;
t12 = q2_dot.^2;
t13 = t3.^2;
t14 = cos(t5);
t15 = m1.*t6.*4.0;
t16 = m2.*t6.*1.2e+1;
t17 = m2.*t6.*t13.*9.0;
t18 = -t17;
t19 = t15+t16+t18;
t20 = 1.0./t19;
f1_disc = [q1+dt.*q1_dot;q2+dt.*q2_dot;q1_dot+(dt.*t20.*(L2.*tau_1.*4.0-L2.*tau_2.*4.0-L1.*t3.*tau_2.*6.0-L1.*L2.*g.*m1.*t2.*2.0-L1.*L2.*g.*m2.*t2.*4.0+L1.*m2.*t4.*t8.*t11.*2.0+L1.*m2.*t4.*t8.*t12.*2.0+L1.*L2.*g.*m2.*t3.*t14.*3.0+L1.*m2.*q1_dot.*q2_dot.*t4.*t8.*4.0+L2.*m2.*t3.*t4.*t6.*t11.*3.0).*3.0)./L2;q2_dot-(dt.*t20.*(m1.*t6.*tau_2.*-4.0-m2.*t6.*tau_2.*1.2e+1+m2.*t8.*tau_1.*4.0-m2.*t8.*tau_2.*4.0+L1.*L2.*m2.*t3.*tau_1.*6.0-L1.*L2.*m2.*t3.*tau_2.*1.2e+1-L1.*g.*t2.*t8.*t10.*4.0+L2.*g.*t6.*t10.*t14.*6.0+L2.*t4.*t7.*t10.*t11.*6.0+L1.*t4.*t9.*t10.*t11.*2.0+L1.*t4.*t9.*t10.*t12.*2.0+t3.*t4.*t6.*t8.*t10.*t11.*6.0+t3.*t4.*t6.*t8.*t10.*t12.*3.0-L1.*g.*m1.*m2.*t2.*t8.*2.0+L2.*g.*m1.*m2.*t6.*t14.*2.0+L2.*m1.*m2.*t4.*t7.*t11.*2.0-L2.*g.*t2.*t3.*t6.*t10.*6.0+L1.*g.*t3.*t8.*t10.*t14.*3.0+L1.*q1_dot.*q2_dot.*t4.*t9.*t10.*4.0-L2.*g.*m1.*m2.*t2.*t3.*t6.*3.0+q1_dot.*q2_dot.*t3.*t4.*t6.*t8.*t10.*6.0).*3.0)./(m2.*t8)];
