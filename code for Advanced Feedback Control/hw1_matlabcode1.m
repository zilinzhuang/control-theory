% system 
A = [-1 0 0; 0 -8 8; 0 0 -2];
B = [1; 0; 1];         % case(d)
C = [1 1 0];           
D = 0;

% get transfer function 
sys = ss(A,B,C,D);
G   = tf(sys);        
disp('G(s) ='); 
G

[num,den] = tfdata(G,'v');   % get num den 


% controller-canonical form
Ac = [0 1 0;
      0 0 1;
     -den(end) -den(end-1) -den(end-2)];  
Bc = [0;0;1];

Cc=[num(end) num(end-1) num(end-2)];

Dc = 0;

sys_c = ss(Ac,Bc,Cc,Dc);

% test
tf(sys_c)