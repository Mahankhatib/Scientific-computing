function yout=rk4singlestep(fun,dt,tk,yk)
    f1=fun(tk,yk);
    f2=fun(tk+dt/2,yk+(dt/2)*f1);
    f3=fun(tk+dt/2,yk+(dt/2)*f2);
    f4=fun(tk+dt,yk+dt*f3);
    yout=yk+(dt/6)*(f1+2*f2+2*f3+f4);
end

function dy=lorentzdyn(t,y,beta,rho,sigma)
    dy(1) = sigma * (y(2) - y(1));
    dy(2) = rho * y(1) - y(2) - y(1) * y(3);
    dy(3) = -beta * y(3) + y(1) * y(2);
    dy=[dy(1);dy(2);dy(3)];
end


%parameters
beta=8/3;
sigma=10;
rho=28;
dt=0.01;
T=15;
timespan=[0:dt:T];
N=length(timespan);
y0=[-8;8;27]; %initial conditions

%simulating by creating a new vector and keeping track by adding yout at
%each time step

Y(:,1)=y0;
yin=y0;

for i=1:timespan(end)/dt
    time=i*dt;
    yout = rk4singlestep(@(t,y) lorentzdyn(t,y,beta,rho,sigma), dt, time, yin);
    Y =[Y yout];
    yin=yout;
end

plot3(Y(1,:),Y(2,:),Y(3,:))






   


