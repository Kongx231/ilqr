function DR1 = calc_Dr12(in1,uy,in3)
%CALC_DR12
%    DR1 = CALC_DR12(IN1,UY,IN3)

%    This function was generated by the Symbolic Math Toolbox version 8.6.
%    12-Nov-2021 01:22:00

e = in3(:,3);
DR1 = reshape([1.0,0.0,0.0,-e],[2,2]);
