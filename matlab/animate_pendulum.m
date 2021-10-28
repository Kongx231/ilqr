function animate_pendulum(states,dt)
bRecord = 1;  
%  bRecord = 0; % Uncomment this to not save a video
if bRecord
    % Define video recording parameters
    Filename = 'pendulum_animation';
    v = VideoWriter(Filename, 'MPEG-4');
    myVideo.Quality = 100;
    open(v);
end

% Define axis window
xmin = -1;
xmax = 1;
ymin = -1;
ymax = 1;

% Draw contact surfaces
x_a = linspace(xmin, xmax,500);
y_a = linspace(ymin, ymax,500);
[X,Y] = meshgrid(x_a,y_a);
a1 = Y;

contour(X,Y,a1,[0,0], 'k'); hold on;

% Create trace of trajectory and particle object
h = animatedline('LineStyle', ':', 'LineWidth', 1.5);
particle = [];
string = [];
spring = [];
% Set up axes
axis equal
axis([xmin xmax ymin ymax])
axis off

% draw
for ii = 1:size(states,1)
    drawnow limitrate
    delete(particle) % Erases previous particle
    delete(string)
    delete(spring)
    particle = scatter(cos(states(ii,1)-pi/2),sin(states(ii,1)-pi/2),500, 'MarkerFaceColor',[1;0;0],'MarkerEdgeColor',[0;0;0]);
    string = line([0,cos(states(ii,1)-pi/2)], [0,sin(states(ii,1)-pi/2)], 'Color', [0;0;0],'LineStyle','-');

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