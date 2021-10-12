function animate_car(states,dt)
bRecord = 1;  
%  bRecord = 0; % Uncomment this to not save a video
if bRecord
    % Define video recording parameters
    Filename = 'car_animation';
    v = VideoWriter(Filename, 'MPEG-4');
    myVideo.Quality = 100;
    open(v);
end

% Define axis window
xmin = -1;
xmax = 2;
ymin = -3;
ymax = 3;

% Draw contact surfaces
x_a = linspace(1.25, xmax,500);
y_a = linspace(ymin, ymax,500);
[X,Y] = meshgrid(x_a,y_a);
a1 = Y-2.25;
a2 = Y-1.75;

contour(X,Y,a1,[0,0], 'k'); hold on;
contour(X,Y,a2,[0,0], 'k'); hold on;

% Create trace of trajectory and particle object
h = animatedline('LineStyle', ':', 'LineWidth', 1.5);
particle = [];
string = [];

% Set up axes
axis equal
axis([xmin xmax ymin ymax])

car_length = 0.5;
% draw
for ii = 1:size(states,1)
    drawnow limitrate
    delete(particle) % Erases previous particle
    delete(string)
    
    string = line([states(ii,1)-car_length*cos(states(ii,4)),states(ii,1)], [states(ii,2)-car_length*sin(states(ii,4)),states(ii,2)], 'Color', [1;0;0],'LineStyle','-','LineWidth',30);

    if bRecord
        frame = getframe(gcf);
        writeVideo(v,frame);
    else
        pause(dt - toc(a)); % waits if drawing frame took less time than anticipated
    end
end

if bRecord
    close(v);
end