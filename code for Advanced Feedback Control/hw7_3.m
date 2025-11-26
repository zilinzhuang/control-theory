
% Problem (c)


clear all; close all; clc;
global P u;

%% System data
A = [ 0  1  0   0;
      0  0  1   0;
      0  0  0   1;
     -1 -12 -10 -16];

B = [0 1;
     1 3;
     0 1;
     1 10];

n = 4;               % state dimension
m = 2;               % input dimension
Q = eye(n);
R = eye(m);

% Initial state 
x0 = [0.5; -0.5; 0.4; -0.3; 0];   % [x(0); J(0)]

%% Reference LQR (for comparison)
[K_LQR, P_LQR, ~] = lqr(A,B,Q,R);
disp('LQR solution for Problem 1(c):');
disp('P_LQR ='); disp(P_LQR);
disp('K_LQR ='); disp(K_LQR);

%% ---------- RL initialization ----------
P = ones(4) ;          % critic
P0=P;
K0 = R \ (B'*P0);         % cause R = I2 = B'*P0
Acl = A - B*K0;           % stability check
eig(Acl)

u = zeros(m,1);          % just for logging if needed

T          = 0.05;       % sample time
Fsamples   = 600;        % number of RL samples
batch_size = 10;         % = number of parameters of P (n(n+1)/2=10)

j   = 0;                 % index inside batch
ch  = 0;                 % how many samples in current batch
X   = [];                % store φ(x_k) in batch
Xpi = [];                % store φ(x_k)-φ(x_{k+1})
Y   = [];                % store cost over [kT,(k+1)T]
before_cost = 0;
after_cost  = 0;

% For plotting convergence (Q2, Q3)
F_RL_hist = norm(R\(B'*P));        % iteration 0
err_hist  = norm(R\(B'*P) - K_LQR);
iter      = 0;

% To store history of K (optional)
KK = [];     % [K(:); time]

for k = 1:Fsamples
    
    j  = j + 1;
    ch = ch + 1;
    
    % ----- this state & φ(x_k) -----
    xk = x0(1:n);
    x1 = xk(1); x2 = xk(2); x3 = xk(3); x4 = xk(4);
    phi_k = [ x1^2;
              x1*x2;
              x1*x3;
              x1*x4;
              x2^2;
              x2*x3;
              x2*x4;
              x3^2;
              x3*x4;
              x4^2 ];
    X(j,:) = phi_k';     % for n=4,the matrix phi_k should be like that,also the sequence should be same as P
    
    if j == 1
        before_cost = xk' * P * xk;
    end
    
    % simulate [0,T] we get x_{k+1}, J_{k+1} -----
    [t_seg, x_seg] = ode45(@odefile_4d, [0 T], x0);
    x_next = x_seg(end,1:n)';   % 4x1
    J_next = x_seg(end,n+1);    % scalar
    J_prev = x0(n+1);
    
    if j == 1
 
        after_cost = (J_next - J_prev) + x_next' * P * x_next;
    end
    
    % φ(x_{k+1})
    x1n = x_next(1); x2n = x_next(2); x3n = x_next(3); x4n = x_next(4);
    phi_next = [ x1n^2;
                 x1n*x2n;
                 x1n*x3n;
                 x1n*x4n;
                 x2n^2;
                 x2n*x3n;
                 x2n*x4n;
                 x3n^2;
                 x3n*x4n;
                 x4n^2 ];
    
    % phi(x_k)-phi(x_{k+1}), and its cost 
    Xpi(j,:) = (phi_k - phi_next)';   % shouldbe1x10
    Y(j,1)   = (J_next - J_prev);     % its a scalar
    
    % prepare the initial value 
    x0 = [x_next; J_next];
    
    % update 
    if mod(j,batch_size) == 0
        
        % Check if the "after_cost" is different from the "before_cost".
        % If it is, we need to update our P matrix.  
        if (abs(after_cost - before_cost) > 1e-5) && (ch == batch_size)
            % update：θ is weights
            weights = pinv(Xpi) * Y;   % 10x1
            %  P (4x4 Symmetric) -
            P = [ weights(1)      weights(2)/2   weights(3)/2   weights(4)/2;
                  weights(2)/2    weights(5)     weights(6)/2   weights(7)/2;
                  weights(3)/2    weights(6)/2   weights(8)     weights(9)/2;
                  weights(4)/2    weights(7)/2   weights(9)/2   weights(10) ];
        end
        
        % update：K = R^{-1} B^T P 
        K = R \ (B' * P);            % 2x4
        iter = iter + 1;
        F_RL_hist(end+1) = norm(K);
        err_hist(end+1)  = norm(K - K_LQR);
        
        % keep the history 
        KK = [KK [K(:); k*T]];       
        
        % reset
        j = 0;
        ch = 0;
        X   = [];
        Xpi = [];
        Y   = [];
        
        % update the before_cost
        x_for_cost = x0(1:n);
        before_cost = x_for_cost' * P * x_for_cost;
    end
end

% finalRL-LQR gain
P_RL = P;
K_RL = R \ (B' * P_RL);

disp('Final RL-LQR critic and gain for 1(c):');
disp('P_RL ='); disp(P_RL);
disp('K_RL ='); disp(K_RL);

t_final = Fsamples * T;

%% LQR vs RL-LQR state response
x0_eval = [0.5; -0.5; 0.4; -0.3];

[t_lqr, x_lqr] = ode45(@(t,x)(A - B*K_LQR)*x, [0 t_final], x0_eval);
[t_rl,  x_rl ] = ode45(@(t,x)(A - B*K_RL )*x, [0 t_final], x0_eval);

figure(1); clf;
for i = 1:n
    subplot(n,1,i);
    plot(t_lqr, x_lqr(:,i), 'r', 'LineWidth', 1.8); hold on;
    plot(t_rl,  x_rl(:,i),  'b--', 'LineWidth', 1.8);
    ylabel(sprintf('x_%d', i));
    grid on;
    if i==1
        legend('LQR','RL-LQR');
    end
end
xlabel('Time (s)');
sgtitle('Q1(c): State responses (LQR vs RL-LQR)');

%%  ||F_RL|| converge?
k_vec = 0:iter;

figure(2); clf;
stem(k_vec, F_RL_hist, 'filled', 'LineWidth', 1.5);
xlabel('RL iteration (batch index)');
ylabel('||F_{RL}||_2');
title('Q2(c): Convergence of ||F_{RL}||');
grid on;

%% ||F_RL - F_LQR|| converge?
figure(3); clf;
stem(k_vec, err_hist, 'filled', 'LineWidth', 1.5);
xlabel('RL iteration (batch index)');
ylabel('||F_{RL} - F_{LQR}||_2');
title('Q3(c): Convergence of ||F_{RL} - F_{LQR}||');
grid on;

%% u(t)for LQR vs RL-LQR 
u_lqr = -(K_LQR * x_lqr')';   % length(t_lqr) x 2
u_rl  = -(K_RL  * x_rl')';

figure(4); clf;
subplot(2,1,1);
plot(t_lqr, u_lqr(:,1), 'r', 'LineWidth', 1.8); hold on;
plot(t_rl,  u_rl(:,1),  'b--', 'LineWidth', 1.8);
ylabel('u_1(t)');
legend('LQR','RL-LQR'); grid on;

subplot(2,1,2);
plot(t_lqr, u_lqr(:,2), 'r', 'LineWidth', 1.8); hold on;
plot(t_rl,  u_rl(:,2),  'b--', 'LineWidth', 1.8);
ylabel('u_2(t)'); xlabel('Time (s)');
title('Q4(c): Control inputs (LQR vs RL-LQR)');
grid on;


function xdot = odefile_4d(t, x)
% x = [x1; x2; x3; x4; cost_integral]

global P u;

A = [ 0  1  0   0;
      0  0  1   0;
      0  0  0   1;
     -1 -12 -10 -16];

B = [0 1;
     1 3;
     0 1;
     1 10];

Q = eye(4);
R = eye(2);

x_state = x(1:4);

% this time's control feedback law u = -R^{-1} B^T P x
u = -(R \ (B' * P * x_state));   % 2x1

xdot      = zeros(5,1);
xdot(1:4) = A * x_state + B * u;
xdot(5)   = x_state' * Q * x_state + u' * R * u;
end
