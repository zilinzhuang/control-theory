
clear; clc; close all;

x0 = [1;1];             % initial condition at t0
dur = 10;               % each experiment runs for 10 s

%% (a) alpha = 0 (LTI), t0 = 5 s
alpha = 0;
A_LTI = [0 1; -5 -1];
B = zeros(2,1); C = eye(2); D = zeros(2,1);

t0a = 5;  tf = t0a + dur;
t  = linspace(t0a, tf, 5001);       % uniform grid
u  = zeros(size(t));                % unforced

sysLTI = ss(A_LTI,B,C,D);
[ya, ta, xa] = lsim(sysLTI, u, t, x0);   % xa(:,1)=x1, xa(:,2)=x2

figure('Name','LTI responses (a)&(b)');
subplot(2,1,1); plot(ta, xa(:,1), 'LineWidth',1.2); grid on;
ylabel('x_1(t)'); title('(a)&(b) \alpha=0,LTI');
subplot(2,1,2); plot(ta, xa(:,2), 'LineWidth',1.2); grid on;
ylabel('x_2(t)'); xlabel('t [s]');

%% (b) alpha = 0 (LTI), t0 = 62 s 
t0b = 62; tf = t0b + dur;
t  = linspace(t0b, tf, 5001);
u  = zeros(size(t));
[yb, tb, xb] = lsim(sysLTI, u, t, x0);

% overlay to verify identical shape
subplot(2,1,1); hold on; plot(tb, yb(:,1), '--', 'LineWidth',1.2);
legend('(a) t_0=5','(b) t_0=62','Location','best');
subplot(2,1,2); hold on; plot(tb, yb(:,2), '--', 'LineWidth',1.2);
legend('(a) t_0=5','(b) t_0=62','Location','best');



%%  (c) alpha = 1 (LTV), t0 = 5 s 

ltvSys = ltvss(@ltvssDataFcn);
t = linspace(t0a, t0a+dur, 5001);            %Total Simulation Time  
u = [];   % no input                  % initial condition
[y,~,x] = lsim(ltvSys,u,t,x0);

    
figure('Name','LTV responses (c)&(d)');
subplot(2,2,1); plot(t, y(:,1), 'LineWidth',1.2); grid on;
ylabel('x_1(t)'); title('(c) \alpha=1,  t\in[5,15] s');
subplot(2,2,2); plot(t, y(:,2), 'LineWidth',1.2); grid on;
ylabel('x_2(t)'); title('(c) \alpha=1,  t\in[5,15] s');xlabel('t [s]');
hold on


%%  (d) alpha = 1 (LTV), t0 = 62 s 

t = linspace(t0b, t0b + dur, 5001);
u = [];   % no input                  % initial condition
[y,~,x] = lsim(ltvSys,u,t,x0);
subplot(2,2,3); hold on; plot(t, y(:,1), '--', 'LineWidth',1.2);
ylabel('x_1(t)'); title('(d) \alpha=1,  t\in[62,72] s');
subplot(2,2,4); hold on; plot(t, y(:,2), '--', 'LineWidth',1.2);
ylabel('x_2(t)'); title('(d) \alpha=1,  t\in[62,72] s');xlabel('t [s]');


function [A,B,C,D,E,dx0,x0,u0,y0,Delays] = ltvssDataFcn(t)
    A = [1*cos(t) 1; -5 -1];
    B = zeros(2,1);
    C = eye(2);
    D = zeros(2,1);
    E = [];
    dx0 = [];
    x0 = [];
    u0 = [];
    y0 = [];
    Delays = [];
end