clear all 


myfun = @(t,y)[ y(2);
                 y(1)*y(2)^2/(1 + y(1)^2) ];   

bcfun  = @(ya,yb)[ ya(1)-0;      
                   yb(1)-0.5 ];  

guess = @(t)[ 0.5*t; 0.5 ];
solinit = bvpinit(linspace(0,1,1000), guess);
sol = bvp4c(myfun, bcfun, solinit);

t = linspace(0,1,1000);
Y = deval(sol,t); x = Y(1,:); v = Y(2,:);
         

x_exact  = sinh(0.5*t);  
v_exact   = 0.5*cosh(0.5*t);



figure; hold on; grid on;
plot(t,x,'LineWidth',2);
plot(t,x_exact,'--','LineWidth',2);
xlabel('t'); ylabel('x(t)');
title('Optimal trajectory x(t)');
legend('BVP solution','sinh(0.5 t)','Location','best');


figure; hold on; grid on;
plot(t,v,'LineWidth',2);
plot(t,v_exact,'--','LineWidth',2);
xlabel('t'); ylabel('dx(t)/dt');
title('The derivative of Optimal trajectory dx(t)/dt');
legend('BVP solution','v-exact','Location','best');


% hw3_4
% a = 0.4812;   
a = asinh(0.5);  
ode = @(T,y) [y(2); y(1)*y(2)^2/(1+y(1)^2)];  

[T,y] = ode45(ode,[0 1],[0; a]);
x_ivp = y(:,1);
x_bvp= x;

figure; hold on; grid on;
plot(T,x_ivp,'LineWidth',2); 
plot(t,x_bvp,'--','LineWidth',2);
xlabel('t'); ylabel('x(t)');
legend('x-IVP','x-BVP','Location','best');
title('IVP vs BVP solution');

fprintf('x_IVP(1)=%.12f,  x_BVP(1)=%.12f,  diff=%.3e\n', ...
        x_ivp(end), x_bvp(end), x_ivp(end)-x_bvp(end));


