%  Problem 1(b)


clear; clc; close all;
global P u;

%% System data 
A = [0 1; -1 -2];
B = [0; 1];
Q = eye(2);
R = 1;

% Initial condition for training: [x1; x2; cost_integral]
x0 = [-1; 2; 0];   % x(0) = [-1; 2], J(0) = 0

%%  Reference LQR (for comparison)
[K_LQR, P_LQR, ~] = lqr(A,B,Q,R);

disp('LQR solution for Problem 1(b):');
disp('P_LQR ='); disp(P_LQR);
disp('K_LQR ='); disp(K_LQR);

%% RL initialization 
P = zeros(2);       
K = (B'*P)/R;       % initial gain (1x2), here = [0 0]

% Training parameters
T          = 0.05;      % sample time [s]
N_samples  = 400;       % number of RL samples
batch_size = 10;        % LS batch size

% For LS
j   = 0;
Xpi = [];
Y   = [];

% For Q2 & Q3: store ||F_RL|| and ||F_RL - F_LQR||
F_RL_hist = norm(K);                % iteration 0
err_hist  = norm(K - K_LQR);        % iteration 0
iter      = 0;                      % LS update index

% keep the traj
t_train = [];
x_train = [];
u_train = [];

before_cost = x0(1:2)' * P * x0(1:2);   % V(x0) before first batch

%%  RL-LQR TRAINING LOOP 
for k = 1:N_samples
    
    j = j + 1;
    
    % current state
    xk = x0(1:2);
    
    % basis phi(x_k) = [x1^2; x1*x2; x2^2]
    phi_k = [ xk(1)^2;
              xk(1)*xk(2);
              xk(2)^2 ];
    
    % simulate system + cost over [0, T]
    [t_seg, x_seg] = ode45(@odefile_2d, [0 T], x0);
    
    % state & cost at t = T
    x_next = x_seg(end,1:2)';   % 2x1
    J_next = x_seg(end,3);
    J_prev = x0(3);
    
    if j == 1
        % cost-to-go after this batch
        after_cost = (J_next - J_prev) + x_next'*P*x_next;
    end
    
    % phi at x_{k+1}
    phi_next = [ x_next(1)^2;
                 x_next(1)*x_next(2);
                 x_next(2)^2 ];
    
    % one row of LS data
    Xpi(j,:) = (phi_k - phi_next)';   % 1x3
    Y(j,1)   = (J_next - J_prev);     % scalar
    
    % store trajectories
    t_train = [t_train; t_seg + (k-1)*T];
    x_train = [x_train; x_seg(:,1:2)];
    u_train = [u_train; u*ones(size(t_seg))];
    
    % update initial condition for next step
    x0 = [x_next; J_next];
    
    %% LS update
    if mod(j, batch_size) == 0
        
        if abs(after_cost - before_cost) > 1e-8
            % weights = [p1; p2; p3]
            weights = pinv(Xpi)*Y;   % 3x1
            
            % reconstruct symmetric P from weights
            % P = [p1, p2/2; p2/2, p3]
            P = [ weights(1)    weights(2)/2;
                  weights(2)/2  weights(3)   ];
        end
        
        % Policy improvement: update K = R^{-1} B^T P
        K = (B'*P)/R;      % 1x2
        
        % record for Q2 & Q3
        iter = iter + 1;
        F_RL_hist(end+1) = norm(K);
        err_hist(end+1)  = norm(K - K_LQR);
        
        % reset batch
        j    = 0;
        Xpi  = [];
        Y    = [];
        
        before_cost = x0(1:2)' * P * x0(1:2);
    end
end

% final learned gain
K_RL = K;
P_RL = P;

disp('Final RL-LQR critic and gain for 1(b):');
disp('P_RL ='); disp(P_RL);
disp('K_RL ='); disp(K_RL);

t_final = N_samples * T;   % training time ≈ evaluation horizon

%% Q1: state response, LQR vs RL-LQR 
x0_eval = [-1; 2];

[t_lqr, x_lqr] = ode45(@(t,x)(A - B*K_LQR)*x, [0 t_final], x0_eval);
[t_rl,  x_rl ] = ode45(@(t,x)(A - B*K_RL )*x, [0 t_final], x0_eval);

figure(1); clf;
subplot(2,1,1);
plot(t_lqr, x_lqr(:,1), 'r', 'LineWidth', 1.8); hold on;
plot(t_rl,  x_rl(:,1),  'b--', 'LineWidth', 1.8);
ylabel('x_1'); grid on;
legend('LQR','RL-LQR');

subplot(2,1,2);
plot(t_lqr, x_lqr(:,2), 'r', 'LineWidth', 1.8); hold on;
plot(t_rl,  x_rl(:,2),  'b--', 'LineWidth', 1.8);
ylabel('x_2'); xlabel('Time (s)'); grid on;
sgtitle('Q1(b): State response with LQR vs RL-LQR');

%%  Q2:convergence of ||F_RL|| 
k_vec = 0:iter;

figure(2); clf;
stem(k_vec, F_RL_hist, 'filled', 'LineWidth', 1.6);
xlabel('RL iteration (LS update index)');
ylabel('||F_{RL}||_2');
title('Q2(b): Convergence of ||F_{RL}||');
grid on;

%%  Q3: convergence of ||F_RL - F_LQR|| 
figure(3); clf;
stem(k_vec, err_hist, 'filled', 'LineWidth', 1.6);
xlabel('RL iteration (LS update index)');
ylabel('||F_{RL} - F_{LQR}||_2');
title('Q3(b): Convergence of ||F_{RL} - F_{LQR}||');
grid on;

%%  Q4: control input u(t), LQR vs RL-LQR 
u_lqr = -(K_LQR * x_lqr')';   % N x 1
u_rl  = -(K_RL  * x_rl')';    % N x 1

figure(4); clf;
plot(t_lqr, u_lqr, 'r', 'LineWidth', 1.8); hold on;
plot(t_rl,  u_rl,  'b--', 'LineWidth', 1.8);
xlabel('Time (s)');
ylabel('u(t)');
legend('LQR','RL-LQR','Location','best');
title('Q4(b): Control input with LQR vs RL-LQR');
grid on;



function xdot = odefile_2d(t, x)
% odefile_2d : dynamics + cost integrator for Problem 1(b)
% x = [x1; x2; cost_integral]

global P u;

A = [0 1; -1 -2];
B = [0; 1];
Q = eye(2);
R = 1;

x_state = x(1:2);

% current control from critic P: u = -R^{-1} B'^P x
u = -(B'*P*x_state)/R;   % scalar

xdot        = zeros(3,1);
xdot(1:2)   = A*x_state + B*u;                 % dx/dt
xdot(3)     = x_state'*Q*x_state + u'*R*u;     % dJ/dt = x'Qx + u'Ru
end
