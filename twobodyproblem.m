function dy=twobody(t,y,G,m)
    % %option1;static earth
    % r=sqrt((y(1)^2)+(y(2)^2)+(y(3)^2));
    % dy(1)=y(4);
    % dy(2)=y(5);
    % dy(3)=y(6);
    % dy(4)=(-G*m/(r^3))*y(1);
    % dy(5)=(-G*m/(r^3))*y(2);
    % dy(6)=(-G*m/(r^3))*y(3);
    % dy=[dy(1);dy(2);dy(3);dy(4);dy(5);dy(6)];

    %option2;moving earth
    vxe=3000;
    vye=3000;
    vze=3000;
    xe=vxe*t;
    ye=vye*t;
    ze=vze*t;

    % Update the position of the moving Earth
    r = sqrt((y(1) - xe)^2 + (y(2) - ye)^2 + (y(3) - ze)^2);
    dy(1) = y(4);
    dy(2) = y(5);
    dy(3) = y(6);
    dy(4) = (-G * m / r^3) * (y(1) - xe);
    dy(5) = (-G * m / r^3) * (y(2) - ye);
    dy(6) = (-G * m / r^3) * (y(3) - ze);
    dy = [dy(1); dy(2); dy(3); dy(4); dy(5); dy(6)];

end

function yout=rk4singlestep(fun,tk,yk,dt)
    f1=fun(tk,yk);
    f2=fun(tk+dt/2,yk+(dt/2)*f1);
    f3=fun(tk+dt/2,yk+(dt/2)*f2);
    f4=fun(tk+dt,yk+dt*f3);
    yout=yk+(dt/6)*(f1+2*f2+2*f3+f4);
end
% Initialize parameters for the simulation
vxe=3000;
vye=3000;
vze=3000;
G=6.67e-11;
m=5.97e24;
t0 = 0; % initial time
y0 = [6771000; 0; 0; 3000; 7672; 2000]; % initial state [x, y, z, vx, vy, vz]
dt = 1; % time step
T = 6000; % end time
timespan=[0:dt:T];
N=length(timespan);
Y=zeros(6,N);
Y(:, 1) = y0; % set initial state
yin=y0;

for i=1:(N-1)
    time=timespan(i);
    yout=rk4singlestep(@(t,y) twobody(t,y,G,m),time,yin,dt);
    Y(:,i+1)=yout;
    yin=yout;
end

%1. Plot the continuous trajectory line
plot3(Y(1,:), Y(2,:), Y(3,:), 'b-', 'LineWidth', 2);
hold on;

%2. Plot the central body (Earth)
plot3(0, 0, 0, 'ko', 'MarkerSize', 12, 'MarkerFaceColor', [0.5, 0.5, 0.5]); 

%3. HIGHLIGHT THE START POINT (First column of Y)
%We use a green circle ('go')
plot3(Y(1,1), Y(2,1), Y(3,1), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
text(Y(1,1), Y(2,1), Y(3,1), '  Start', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'g');

% 4. HIGHLIGHT THE END POINT (Last column of Y)
% We use a square marker ('rs')
plot3(Y(1,end), Y(2,end), Y(3,end), 'rs', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
text(Y(1,end), Y(2,end), Y(3,end), '  End', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'r');




%plotting the earth trajectory
XE=vxe*timespan;
YE=vye*timespan;
ZE=vze*timespan;
plot3(XE,YE,ZE,"-g",LineWidth=2);
plot3(XE(1), YE(1), ZE(1), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', [0.5, 0.5, 0.5]);
text(XE(1), YE(1), ZE(1), '  Earth Start', 'FontSize', 10, 'FontWeight', 'bold');

plot3(XE(end), YE(end), ZE(end), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', [0.5, 0.5, 0.5]);
text(XE(end), YE(end), ZE(end), '  Earth End', 'FontSize', 10, 'FontWeight', 'bold');

% 5. Clean up the view
grid on;
xlabel('X Position'); ylabel('Y Position');
title('Orbit Simulation with Start and End Points');
legend("Satellite Trajectory","Earth Trajectory")

% plot(Y(1,:),Y(2,:));
% xlabel("X position");
% ylabel("Y position");
% title("2D trajectory")

