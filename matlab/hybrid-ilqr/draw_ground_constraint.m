function draw_ground_constraint()
% Define axis window
xmin = -5;
xmax = 5;
ymin = -1;
ymax = 6;

% Draw contact surfaces
x_a = linspace(xmin, xmax,500);
y_a = linspace(ymin, ymax,500);
[X,Y] = meshgrid(x_a,y_a);
a1 = Y;
% coeff = calc_constraint_coeff();
% a1 = coeff(1)*X+coeff(2)*Y - ones(size(Y))*coeff(3);
% Only want 1 contact mode for now
contour(X,Y,a1,[0,0], 'k'); hold on;
end