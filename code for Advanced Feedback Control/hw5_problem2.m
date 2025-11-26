% problem 2/b/i
A = [0 1 0; 0 0 1; 0 -4 -5];
B = [0;0;1];
Q = [8 6 0; 6 6 0; 0 0 4];
R = 1;
x0 = [2;0;-2];


eigA = eig(A)                  
[K,S,~] = lqr(A,B,Q,R);        
Acl = A - B*K;
eigAcl = eig(Acl)              

sysCL = ss(Acl, zeros(3,1), eye(3), zeros(3,1));
t = linspace(0,12,12001);     
[y,~,~] = lsim(sysCL, zeros(numel(t),1), t, x0);  
x = y;
u = -K*x.'; u = u.';           

figure; plot(t, x); grid on; xlabel t; ylabel('states'); legend x1 x2 x3
title('Closed-loop states');
figure; plot(t, u); grid on; xlabel t; ylabel u; title('Control input');

% problem 2/b/ii
thr = 1e-3;
mask = max([abs(x), abs(u)],[],2) < thr;
idx  = find(mask,1,'first');
t_allzero = t(idx);

J = trapz(t(1:idx), sum((x(1:idx,:)*(Q)).*x(1:idx,:),2) + (u(1:idx,:).*u(1:idx,:))*R);

fprintf('t_settle ≈ %.4f s,   energy J ≈ %.6f\n', t_allzero, J);

% problem 2/b/iii
QQ = [1 2 4 10 20];
output1 = [];
for a = QQ
    [K,S] = lqr(A,B,a*Q,R);
    Acl = A - B*K;
    lam = eig(Acl);
    rmax = max(abs(lam)); rmin = min(abs(lam)); con_mtrx = rmax/rmin;
    output1 = [output1; a, rmax, rmin, con_mtrx];
end
disp(array2table(output1, 'VariableNames',{'Q','abs_lambda_max','abs_lambda_min','con_mtrx'}));


% problem 2/b/iv
RR = [1 2 4 10 20];
output2 = [];
for b = RR
    Rb = b*R;                  % R=0.5*b
    [K,S] = lqr(A,B,Q,Rb);
    Acl = A - B*K;
    lam = eig(Acl);
    rmax = max(abs(lam)); rmin = min(abs(lam)); con_mtrx = rmax/rmin;
    output2 = [output2; Rb(1), rmax, rmin, con_mtrx];
end
disp(array2table(output2, 'VariableNames',{'R','abs_lambda_max','abs_lambda_min','con_mtrx'}));

