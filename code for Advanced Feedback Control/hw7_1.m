%  Problem 1(a)


clear; clc; close all;
global P u;

%% given plant data 
A = 2;
B = 1;
Q = 1;
R = 1;

% setup: Initial condition of state and cost-integrator
x0 = [1; 0];          % [x; J], x(0) = 1, J(0) = 0

%% Reference value of LQR 
[K_LQR, P_LQR, ~] = lqr(A,B,Q,R);    % P_LQR is the Riccati solution

disp('True LQR solution:');
disp(['P_LQR = ', num2str(P_LQR)]);
disp(['K_LQR = ', num2str(K_LQR)]);

%% ----- RL initial critic/actor -----
P  = 3;               % initial critic (choose P0 > 2 for stability)
K0 = B'*P/R;          % initial K (actor)

% RL sampling parameters
T          = 0.05;    % sample time [s]
N_samples  = 200;     % number of RL time-steps
batch_size = 6;       % batch length for least-squares

% For batch LS
j   = 0;
Xpi = [];
Y   = [];

% Histories for Questions 2 and 3
F_RL_hist = abs(K0);          % ||F_RL||, start from initial K0
err_hist  = abs(K0 - K_LQR);  % ||F_RL - F_LQR||
iter      = 0;                % RL iteration index

% store trajectories during learning
t_train = [];
x_train = [];
u_train = [];

before_cost = P * x0(1)^2;    % V(x0) with initial critic

%% RL–LQR learning loop
for k = 1:N_samples
    
    j = j + 1;
    
    % Basis function at the beginning of the interval:
    % phi(x) = x^2  since V(x)=P x^2
    phi_k = x0(1)^2;
    
    % Simulate plant for one sample period with current critic P
    [t,x] = ode45(@odefile_scalar, [0 T], x0);
    x1    = x(end,1);          % state at t = (k+1)T
    
    if j == 1
        % "after" cost used to decide if we update P
        after_cost = (x(end,2) - x0(2)) + P * x1^2;
    end
    
    % Collect data for LS: phi(x_k) - phi(x_{k+1}) = integral cost
    Xpi(j,1) = phi_k - x1^2;
    Y(j,1)   = x(end,2) - x0(2);   % ∫(x^2 + u^2)dt over this interval
    
    % Store training trajectory (not required by HW, but useful)
    t_train = [t_train; t + (k-1)*T];
    x_train = [x_train; x(:,1)];
    u_train = [u_train; u*ones(size(t))];
    
    % Prepare initial condition for next step
    x0 = [x1; x(end,2)];
    
    %% the update of P
    if mod(j, batch_size) == 0
        
        if abs(after_cost - before_cost) > 1e-6
            theta = pinv(Xpi)*Y;   % critic parameter
            P     = theta(1);      % update critic
        end
        
        % Policy improvement actor update
        K = B'*P/R;
        
        % Store for Q2 & Q3 one point per LS update
        iter = iter + 1;
        F_RL_hist(end+1) = abs(K);         % ||F_RL||
        err_hist(end+1)  = abs(K - K_LQR); % ||F_RL - F_LQR||
        
        % Reset batch buffers
        j    = 0;
        Xpi  = [];
        Y    = [];
        
        % Reset before_cost for the next batch
        before_cost = P * x0(1)^2;
    end
end

% Final learned RL-LQR gain
K_RL = K;
P_RL = P;

disp('Final RL critic and gain:');
disp(['P_RL = ', num2str(P_RL)]);
disp(['K_RL = ', num2str(K_RL)]);

t_final = N_samples * T;   % total training time (also used for evaluation)

%%  Question 1: state response 
% Closed-loop response x(t) with final RL-LQR vs classical LQR

x0_eval = 1;

% RL-LQR (using final K_RL)
[t_rl, x_rl] = ode45(@(t,x) (A - B*K_RL)*x, [0 t_final], x0_eval);

% Classical LQR
[t_lqr, x_lqr] = ode45(@(t,x) (A - B*K_LQR)*x, [0 t_final], x0_eval);

figure(1); clf;
plot(t_lqr, x_lqr, 'r', 'LineWidth', 2); hold on;
plot(t_rl,  x_rl,  'b--', 'LineWidth', 2);
xlabel('Time (s)'); ylabel('x(t)');
legend('LQR', 'RL-LQR', 'Location', 'best');
title('Q1: State response with LQR vs RL-LQR');
grid on;

%% = Question 2: convergence of ||F_RL|| by iterations 
k_vec = 0:iter;   % iteration index (0 = initial K0)

figure(2); clf;
stem(k_vec, F_RL_hist, 'filled', 'LineWidth', 1.5);
xlabel('RL iteration (LS update index)');
ylabel('||F_{RL}||');
title('Q2: Convergence of ||F_{RL}||');
grid on;

%% = Question 3: convergence of ||F_RL - F_LQR||
figure(3); clf;
stem(k_vec, err_hist, 'filled', 'LineWidth', 1.5);
xlabel('RL iteration (LS update index)');
ylabel('||F_{RL} - F_{LQR}||');
title('Q3: Convergence of ||F_{RL} - F_{LQR}||');
grid on;

%% = Question 4: control input u(t) with RL-LQR vs LQR 
u_lqr = -K_LQR * x_lqr;
u_rl  = -K_RL  * x_rl;

figure(4); clf;
plot(t_lqr, u_lqr, 'r', 'LineWidth', 2); hold on;
plot(t_rl,  u_rl,  'b--', 'LineWidth', 2);
xlabel('Time (s)'); ylabel('u(t)');
legend('LQR', 'RL-LQR', 'Location', 'best');
title('Q4: Control signal with LQR vs RL-LQR');
grid on;

% the real system 
function xdot = odefile_scalar(t, x)
% odefile_scalar : dynamics + cost-integral for Problem 1(a)
% x = [state; cost_integral]

global P u;

A = 2;
B = 1;
Q = 1;
R = 1;

% Current control law from critic P (actor)
u = -(B'*P/R) * x(1);   % scalar control

% System dynamics + cost integrator
xdot = [ A*x(1) + B*u;          % dx/dt
         x(1)^2*Q + u^2*R ];    % dJ/dt = x^2 + u^2
end
