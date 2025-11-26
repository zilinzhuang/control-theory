
clear; clc; close all;
xstar = [0.1; 0.3; 0.5];
c12 = cos(xstar(2)-xstar(1)); c13 = cos(xstar(3)-xstar(1)); c23 = c12;
J = [-(c12+c13)   c12         c13 ;
      c12        -(c12+c23)   c23 ;
      c13         c23        -(c13+c23)];
A = [zeros(3) eye(3); J zeros(3)];
B = [zeros(3); eye(3)];

C = [1 0 0 0 0 0;
     0 0 0 1 0 0;
     0 1 0 0 0 0;
     0 0 0 0 1 0];

%%  LQR 
Lg = [ 2 -1 -1; -1  2 -1; -1 -1  2];        
Qx = Lg;  Qv = eye(3);
                              
Q  = blkdiag(Qx , Qv);
R  = 10*eye(3);
[K,~,~] = lqr(A,B,Q,R);
Acl = A - B*K;

%%  Kalman filter
G = eye(6);
W1 = eye(6);
W2 = 2*eye(4);

[S, P, Eest] = lqe(A, G, C, W1, W2);   
disp('Observer poles (A - S*C):'); disp(eig(A - S*C).');

%%  LQG 
Aaug = [A,        -B*K;
        S*C,  A - B*K - S*C];
Baug = [G,  zeros(6,4);     
        zeros(6,6),  S];     

sysLQG = ss(Aaug, Baug, eye(12), zeros(12,10));

%% simulation and noises 
Tend =20;  t = linspace(0,Tend,15000); 

sigma_w = 0.06;                        
w = sigma_w * randn(numel(t), 6);


z0 = [0.30; -0.15; -0.15; 0; 0; 0];    
zh0= zeros(6,1);                       
x0 = [z0; zh0];
z_clean = initial(ss(Aaug,[],eye(12),[]), x0, t);  

y_clean = (C * z_clean(:,1:6).').';    
SNR_m = 25;                     
y_meas = awgn(y_clean, SNR_m, 'measured');   

v = y_meas - y_clean;
u_noise = [w, v];
zLQG = lsim(sysLQG, u_noise, t, x0);     
z_plant = zLQG(:,1:6);
zhat = zLQG(:,7:12);
u_lqg = -(K * zhat.').';


z_full = initial(ss(Acl,[],eye(6),[]), z0, t); 
u_full = -(K*z_full.').';

figure('Color','w');
subplot(3,1,1); plot(t,z_full(:,1),'k-', t,z_plant(:,1),'b--','LineWidth',1.3);
grid on; ylabel('\Delta x_1'); legend('Full LQR','LQG (Kalman)','Location','northeast');
title('Closed-loop positions');
subplot(3,1,2); plot(t,z_full(:,2),'k-', t,z_plant(:,2),'b--','LineWidth',1.3); grid on; ylabel('\Delta x_2');
subplot(3,1,3); plot(t,z_full(:,3),'k-', t,z_plant(:,3),'b--','LineWidth',1.3); grid on; ylabel('\Delta x_3'); xlabel('t (s)');

figure('Color','w');
plot(t, u_full(:,1),'k-', t,u_lqg(:,1),'b--', ...
     t, u_full(:,2),'k-', t,u_lqg(:,2),'b--', ...
     t, u_full(:,3),'k-', t,u_lqg(:,3),'b--', 'LineWidth',1.0);
grid on; xlabel('t (s)'); ylabel('u_i');
title('Control inputs');
legend('u_1 full','u_1 LQG','u_2 full','u_2 LQG','u_3 full','u_3 LQG');


fprintf('rank(obsv(A,C)) = %d (expect 6)\n', rank(obsv(A,C)));
fprintf('eig(A-BK):\n'); disp(eig(Acl).');
fprintf('eig(A-SC):\n'); disp(eig(A - S*C).');
