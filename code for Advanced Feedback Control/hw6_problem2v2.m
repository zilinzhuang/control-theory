%% observer-based LQR with error-state augmentation (no steady-state bias)
clear; clc; close all;

%% model
xstar = [0.1; 0.3; 0.5];
c12 = cos(xstar(2)-xstar(1)); c13 = cos(xstar(3)-xstar(1)); c23 = cos(xstar(3)-xstar(2));
J = [-(c12+c13) c12 c13; c12 -(c12+c23) c23; c13 c23 -(c13+c23)];
A = [zeros(3) eye(3); J zeros(3)];
B = [zeros(3); eye(3)];

%% LQR 
L  = [ 2 -1 -1; -1  2 -1; -1 -1  2];
Qx = 2*L;  Qv = eye(3);
delta = 1e-3;           
Qx = Qx + delta*eye(3);
Q  = blkdiag(Qx, Qv);
R  = 10*eye(3);
[K,~,~] = lqr(A,B,Q,R);
Acl = A - B*K;

%% measurements

%   C = [1 0 0 0 0 0;
%        0 1 0 0 0 0;
%        0 0 0 1 0 0;
%        0 0 0 0 1 0];
C = [1 0 0 0 0 0;
     0 0 0 1 0 0;
     0 1 0 0 0 0;
     0 0 0 0 1 0];

p_obs = [-2 -2.2 -2.4 -2.6 -2.8 -3.0];
S = place(A',C',p_obs)';    % A - S*C 

%%  x_aug = [z; e],  e = z - zhat
Aee = [Acl,        B*K;   % zdot = (A-BK) z + B K e
       zeros(6),  (A - S*C)];     % edot = (A - S C) e

%% compare
Tend = 15;  t = linspace(0,Tend,12001);
z0   = [0.30; -0.15; -0.15; 0; 0; 0];   
zh0  = zeros(6,1);                      
e0   = z0 - zh0;                       

% full time LQR
sys_full = ss(Acl, [], eye(6), []);
z_full   = initial(sys_full, z0, t);
u_full   = -(K*z_full.').';

% obeserve-LQR
sys_err  = ss(Aee, [], eye(12), []);
xe       = initial(sys_err, [z0; e0], t);
z_obs    = xe(:,1:6);         
e_obs    = xe(:,7:12);        
zhat     = z_obs - e_obs;     
u_obs    = -(K*zhat.').';

%% -
max_diff = max(abs(z_full(:) - z_obs(:)));
fprintf('max |z_full - z_obs| = %.3e ( 1e-10 ~ 1e-12)\n', max_diff);

%% --- 画图 ---
figure('Color','w');
subplot(3,1,1); plot(t,z_full(:,1),'k-',t,z_obs(:,1),'b--','LineWidth',1.3); grid on; ylabel('\Delta x_1');
legend('full-state LQR','observer-based LQR');
title('Closed-loop positions: full vs observer-based (error-state implementation)');

subplot(3,1,2); plot(t,z_full(:,2),'k-',t,z_obs(:,2),'b--','LineWidth',1.3); grid on; ylabel('\Delta x_2');
subplot(3,1,3); plot(t,z_full(:,3),'k-',t,z_obs(:,3),'b--','LineWidth',1.3); grid on; ylabel('\Delta x_3'); xlabel('t (s)');

figure('Color','w');
plot(t,u_full(:,1),'k-', t,u_obs(:,1),'b--', ...
     t,u_full(:,2),'k-', t,u_obs(:,2),'b--', ...
     t,u_full(:,3),'k-', t,u_obs(:,3),'b--', 'LineWidth',1.0);
grid on; xlabel('t (s)'); ylabel('u_i');
title('Control inputs: full-state vs observer-based');
legend('u_1 full','u_1 obs','u_2 full','u_2 obs','u_3 full','u_3 obs');
