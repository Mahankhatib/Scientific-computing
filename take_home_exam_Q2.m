
%loading the trasfer function
my_tf=tf(sys);

%bode plots
%figure;
%grid on;
%bode(my_tf);

zero(my_tf)
pole(my_tf)  %all poles on the left half of (complex) plane
figure;
pzmap(my_tf)  %visualizing the poles
%figure;
%step(my_tf);   %settling over a finite value in finite time--> stable
%figure;
%impulse(my_tf)

%figure;
%margin(my_tf)
figure;
%rlocus(my_tf)

my_tf;



