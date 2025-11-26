%% Problem b
clear; clc; close all;
xstar = [0.1; 0.3; 0.5];                 
c12 = cos(xstar(2)-xstar(1));
c13 = cos(xstar(3)-xstar(1));
c23 = cos(xstar(3)-xstar(2));

JJ = [-(c12+c13)   c12        c13; 
      c12        -(c12+c23)  c23;
      c13         c23       -(c13+c23)];
A = [zeros(3) eye(3);
     JJ        -eye(3)];
B = [zeros(3); eye(3)];

%% LQR calculation

Qx  = [ 2 -1 -1;
     -1  2 -1;
     -1 -1  2];             
Qv  = eye(3);           
Q   = blkdiag(Qx, Qv);  
R   = 10*eye(3);        
[K, S, Ecl] = lqr(A,B,Q,R);   

% important:verify the condition
N  = zeros(6,3);

% 1) [Q N; N'' R] must be PSD 
Qeff = Q - N*(R\N');      % here equals Q
Mblk = [Q N; N' R];
Mblk = (Mblk+Mblk')/2;   
eigM = eig(Mblk);
tolPSD = 1e-9;
ok2 = all(eigM >= -tolPSD);
if ok2
    fprintf(' [Q N; N'' R] PSD?  TRUE   (min eig = %.3e)\n', min(eigM));
else
    fprintf('[Q N; N'' R] PSD?  FALSE   (min eig = %.3e)\n', min(eigM));
end

% 2) (Qeff, Aeff) has no unobservable mode on the imaginary axis

Aeff = A;
Qeff = (Qeff+Qeff')/2;    
Ceff = real(sqrtm(Qeff)); 

lam   = eig(Aeff);
tolRe = 1e-8;
tolRK = 1e-8;
n = size(Aeff,1);
idx = find(abs(real(lam)) < tolRe);

viol = [];              
for kk = 1:numel(idx)
    l = lam(idx(kk));
    Rk = [l*eye(n) - Aeff; Ceff];  
    if rank(Rk, tolRK) < n
        viol(end+1,1) = l;
    end
end
if isempty(viol)
    fprintf('No unobservable jω modes?  TRUE\n');
else
    fprintf('No unobservable jω modes?  FALSE\n');
    disp(' Unobservable eigenvalues on jω :');
    disp(viol.');         
end




% delta = 1e-6;                 
% v = ones(3,1);
% Qx = Qx + delta*(v*v.'); 
% Q  = blkdiag(Qx, Qv);
% 
% 
% [K, S, Ecl] = lqr(A,B,Q,R);   
Acl = A - B*K;

K   
S

disp('Openloop eig:'); disp(eig(A).');
disp('Closedloop eig:'); disp(eig(Acl).');


Tend = 15;                       
t = linspace(0, Tend, 12001);     
z0 = [0.30; -0.15; -0.15; 0; 0; 0];   


sysOPENL = ss(A, zeros(6,3), eye(6), zeros(6,3));
x_ol = initial(sysOPENL, z0, t);     


sysCLOSEL = ss(Acl, zeros(6,3), eye(6), zeros(6,3));
x_cl = initial(sysCLOSEL, z0, t);
u_cl = -(K * x_cl.').';          


figure('Color','w'); 
plot(t, x_ol(:,1), 'LineWidth',1.6); hold on;
plot(t, x_ol(:,2), 'LineWidth',1.6);
plot(t, x_ol(:,3), 'LineWidth',1.6);
grid on; xlabel('t (s)'); ylabel('\Delta x_i');
title('Open-loop positions');
legend('\Delta x_1','\Delta x_2','\Delta x_3');

figure('Color','w'); 
plot(t, x_cl(:,1), 'LineWidth',1.6); hold on;
plot(t, x_cl(:,2), 'LineWidth',1.6);
plot(t, x_cl(:,3), 'LineWidth',1.6);
grid on; xlabel('t (s)'); ylabel('\Delta x_i');
title('Closed-loop positions');
legend('\Delta x_1','\Delta x_2','\Delta x_3');


figure('Color','w');
plot(t, u_cl(:,1), 'LineWidth',1.4); hold on;
plot(t, u_cl(:,2), 'LineWidth',1.4);
plot(t, u_cl(:,3), 'LineWidth',1.4);
grid on; xlabel('t (s)'); ylabel('\Delta u_i');
title('Closed-loop control inputs');
legend('u_1','u_2','u_3');


%% Problem c
H = [ A           B*(R\B') ;
     Q             -A.'   ];


lam = eig(H)   



[~,idx] = sort(real(lam));
lam_sorted = lam(idx);

sysH = ss(H, zeros(size(H,1),1), eye(size(H,1)), zeros(size(H,1),1));
figure('Color','w'); pzmap(sysH); grid on; axis equal;
title('Hamiltonian eigenvalues');



