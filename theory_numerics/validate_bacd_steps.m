function validate_bacd_steps()
%VALIDATE_BACD_STEPS Compare BACD slopes at h=0.02 and h=0.01.

rng(1);
m = 3; n = 2; d = m*n;
B = randn(d);
A = B*B.'/d + 0.8*eye(d);
X0 = 0.4*randn(d,1);
P0 = 0.3*randn(d,1);
R0 = zeros(m,1);
C0 = zeros(n,1);
alpha = 1; mu = 1; T = 3e4;
windows = [1e2 1e3; 1e3 1e4; 1e4 3e4];

for eps_stab = [0 0.1]
    slopes = zeros(2,size(windows,1));
    hs = [0.02 0.01];
    for q = 1:numel(hs)
        [t,E] = bacd_trajectory(A,m,n,alpha,mu,eps_stab,T,hs(q), ...
                                X0,P0,R0,C0);
        for k = 1:size(windows,1)
            take = t >= windows(k,1) & t <= windows(k,2) & E > 0;
            fit = polyfit(log(t(take)),log(E(take)),1);
            slopes(q,k) = fit(1);
        end
    end

    fprintf('\nepsilon_stab = %g\n',eps_stab);
    for k = 1:size(windows,1)
        fprintf('[%.0e,%.0e]  h=.02: %+.9f  h=.01: %+.9f  change: %.9g\n', ...
            windows(k,1),windows(k,2),slopes(1,k),slopes(2,k), ...
            abs(slopes(1,k)-slopes(2,k)));
    end
end
end

function [tout,Eout] = bacd_trajectory(A,m,n,alpha,mu,eps_stab,T,h,X,P,R,C)
ea = exp(-alpha*h);
cf = (1-ea)/(mu*alpha);
sample_times = logspace(0,log10(T),1200).';
tout = zeros(size(sample_times));
Eout = zeros(size(sample_times));
next_sample = 1;
nsteps = round(T/h);

for step = 1:nsteps
    % B and A
    P = P - h*(A*X);
    X = X + h*P;

    % C: first half damping, exact factor update, second half damping
    Pm = reshape(P,m,n);
    den = sum(R) + eps_stab;
    if den > 0
        Pm = Pm .* exp(-0.5*h*(R*C.')/den);
    end
    R = ea*R + cf*sum(Pm.^2,2);
    C = ea*C + cf*sum(Pm.^2,1).';
    den = sum(R) + eps_stab;
    if den > 0
        Pm = Pm .* exp(-0.5*h*(R*C.')/den);
    end
    P = Pm(:);

    t = step*h;
    while next_sample <= numel(sample_times) && t >= sample_times(next_sample)
        tout(next_sample) = t;
        Eout(next_sample) = 0.5*(X.'*A*X + P.'*P);
        next_sample = next_sample + 1;
    end
end

tout = tout(1:next_sample-1);
Eout = Eout(1:next_sample-1);
end
