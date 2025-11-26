clear all

solinit = bvpinit(linspace(0,1,1000), @guess);
sol = bvp4c(@myfun, @bcfun, solinit);


t = linspace(0,1,1000);
Y = deval(sol, t);
x1 = Y(1,:);  m1 = Y(2,:);
x2 = Y(3,:);  m2 = Y(4,:);
x3 = Y(5,:);  m3 = Y(6,:);
x4 = Y(7,:);  m4 = Y(8,:);

figure;
plot(t,x1,'-','LineWidth',2); hold on;
plot(t,x2,'--','LineWidth',2);
plot(t,x3,'-.','LineWidth',2);
plot(t,x4,':','LineWidth',2);
grid on; xlabel('t'); ylabel('x_i^*(t)');
title('Optimal trajectory');
legend('x_1^*','x_2^*','x_3^*','x_4^*','Location','best');

% F = m1.^2 + m2.^2 + m3.^2 + m4.^2 + 2*x1.*x2.*x3.*x4.*sin(t);

    function dydt = myfun(t,y)
      
        x1 = y(1); m1 = y(2);
        x2 = y(3); m2 = y(4);
        x3 = y(5); m3 = y(6);
        x4 = y(7); m4 = y(8);
        dydt = zeros(8,1);
        dydt(1) = m1;
        dydt(2) = x2*x3*x4*sin(t);
        dydt(3) = m2;
        dydt(4) = x1*x3*x4*sin(t);
        dydt(5) = m3;
        dydt(6) = x1*x2*x4*sin(t);
        dydt(7) = m4;
        dydt(8) = x1*x2*x3*sin(t);
    end

    function res = bcfun(ya,yb)

        res = [ ya(1)-1; ya(3)-1; ya(5)-1; ya(7)-1;   % t=0
                yb(1)-pi/2; yb(3)-pi/4; yb(5)-3; yb(7)-4 ]; % t=1
    end

    function y0 = guess(t)

        x1g = 1 + (pi/2 - 1)*t;   m1g = (pi/2 - 1);
        x2g = 1 + (pi/4 - 1)*t;   m2g = (pi/4 - 1);
        x3g = 1 + (3- 1)*t;   m3g = (3    - 1);
        x4g = 1 + (4- 1)*t;   m4g = (4    - 1);
        y0 = [x1g; m1g; x2g; m2g; x3g; m3g; x4g; m4g];
    end





