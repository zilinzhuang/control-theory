
alpha = sqrt(60);
constant1= 8*( -exp(-alpha) +exp(alpha)        )
A=(33-9*exp(-alpha)) / (constant1 )
B= (-33+9*exp(alpha)) / (constant1 )

xstar  = @(t) A*exp(alpha*t) + B*exp(-alpha*t) - 1/8;
xdstar = @(t) alpha*(A*exp(alpha*t) - B*exp(-alpha*t));

F = @(t,x,xd) (xd.^2)/5 + 4*x.*xd + 12*x.^2 + 3*x;


fplot(xstar,[0,1],'LineWidth',1.6); grid on;
xlabel('t'); ylabel('x^*(t)'); title('Optimal trajectory x^*(t)');


figure; fplot(xdstar,[0,1],'LineWidth',1.6); grid on;
xlabel('t'); ylabel('\it\dot{x}^*(t)'); title('Derivative of the optimal trajectory');


Jstar = integral(@(t) F(t,xstar(t),xdstar(t)), 0, 1);
fprintf('J* (optimal) = %.6f\n', Jstar);


xalt  = @(t) 1 + 3*t;
xdalt = @(t) 3 + 0*t;
Jalt  = integral(@(t) F(t,xalt(t),xdalt(t)), 0, 1);
fprintf('J for x(t)=1+3t     = %.6f\n', Jalt);



xalt  = @(t) 2*t.^2+1+sin(pi*t/2);
xdalt = @(t) 4*t + (pi/2)*cos(pi*t/2);
Jalt  = integral(@(t) F(t,xalt(t),xdalt(t)), 0, 1);
fprintf('J for x(t)=2t^2+1+sin(pi*t/2)     = %.6f\n', Jalt);



