function validate_rates()
%VALIDATE_RATES  Numerical validation of the gamma=0 algebraic rate relations.
%
%   Asymptotics of the gamma=0 decay (exponents identified by formal averaging):
%       eps = 0 :  E(t) ~ alpha*mu / t
%       eps > 0 :  E(t) ~ alpha*mu*sqrt(eps/(2t))
%   with crossover between the two at E ~ alpha*mu*eps.
%
%   The tests below check (A) the exponents, (B),(C) the predicted dependence
%   on alpha, mu and eps of the prefactors, and (D) the crossover location.

rng(0);
opts = odeset('RelTol',1e-10,'AbsTol',1e-12);

fprintf('\n=== A. exponents, dense SPD Hessian, m=3 n=2, alpha=mu=1 ===\n');
[m,n] = deal(3,2);  A = spdmat(m*n);
for eps = [0 0.1]
    [t,E,~] = trajectory(A,m,n,1,1,eps,3e4,opts);
    fprintf('  eps=%-5g  ', eps);
    for w = [1e2 1e3; 1e3 1e4; 1e4 3e4].'
        fprintf('slope[%.0e,%.0e]=%+.3f  ', w(1), w(2), loglogslope(t,E,w(1),w(2)));
    end
    fprintf('(predicted %+.1f)\n', -1 + 0.5*(eps>0));
end

fprintf('\n=== B. eps=0: is  t*E(t)/(alpha*mu)  constant across alpha,mu? ===\n');
fprintf('   alpha     mu      t*E/(alpha*mu) at t=1e4\n');
vB = [];
for al = [0.5 1 2]
    for mu = [0.5 1 2]
        [t,E,~] = trajectory(A,m,n,al,mu,0,1e4,opts);
        v = interp1(t,E,1e4)*1e4/(al*mu);
        vB(end+1) = v; %#ok<AGROW>
        fprintf('   %-8.2f %-8.2f %8.4f\n', al, mu, v);
    end
end
fprintf('   -> mean %.3f, spread max/min = %.3f  (alpha,mu each vary by 4x)\n', ...
        mean(vB), max(vB)/min(vB));

fprintf('\n=== C. eps>0: is  E(t)*sqrt(t)/(alpha*mu*sqrt(eps/2))  constant? ===\n');
fprintf('   alpha     mu      eps      ratio at t=1e4\n');
vC = [];
for al = [0.5 1]
    for mu = [0.5 1]
        for eps = [0.05 0.2]
            [t,E,~] = trajectory(A,m,n,al,mu,eps,1e4,opts);
            v = interp1(t,E,1e4)*sqrt(1e4)/(al*mu*sqrt(eps/2));
            vC(end+1) = v; %#ok<AGROW>
            fprintf('   %-8.2f %-8.2f %-8.3g %8.4f\n', al, mu, eps, v);
        end
    end
end
fprintf('   -> mean %.3f, spread max/min = %.3f\n', mean(vC), max(vC)/min(vC));

fprintf('\n=== D. crossover S ~ eps occurs at E ~ alpha*mu*eps ===\n');
fprintf('   alpha     mu      eps      E at S=eps    /(alpha*mu*eps)\n');
vD = [];
for al = [0.5 1]
    for mu = [0.5 1]
        for eps = [0.05 0.2]
            [t,E,S] = trajectory(A,m,n,al,mu,eps,1e4,opts);
            k = find(S >= eps, 1, 'last');   % S(0)=0, so use the downward crossing
            if isempty(k) || k == numel(S)
                fprintf('   %-8.2f %-8.2f %-8.3g  (no downward crossing in [0,T])\n', al, mu, eps);
                continue;
            end
            vD(end+1) = E(k)/(al*mu*eps); %#ok<AGROW>
            fprintf('   %-8.2f %-8.2f %-8.3g %10.3e  %8.4f\n', ...
                    al, mu, eps, E(k), E(k)/(al*mu*eps));
        end
    end
end

if ~isempty(vD)
    fprintf('   -> mean %.3f, spread max/min = %.3f\n', mean(vD), max(vD)/min(vD));
end

fprintf('\n=== F. scalar case m=n=1 (this is iKFAD): prefactors ===\n');
A1 = 1;                      % f(x) = x^2/2
[t,E,~] = trajectory(A1,1,1,1,1,0,1e5,opts);
fprintf('   eps=0    : t*E/(alpha*mu) at t=1e5 = %.4f   (formal prediction 1)\n', ...
        interp1(t,E,1e5)*1e5);
[t,E,~] = trajectory(A1,1,1,1,1,0.1,1e5,opts);
fprintf('   eps=0.1  : E*sqrt(t)/(alpha*mu*sqrt(eps/2)) at t=1e5 = %.4f   (prediction 1)\n', ...
        interp1(t,E,1e5)*sqrt(1e5)/sqrt(0.1/2));

fprintf('\n=== E. exponents are independent of the matrix dimensions ===\n');
for dims = [2 2; 4 3; 6 5].'
    m2 = dims(1); n2 = dims(2); A2 = spdmat(m2*n2);
    for eps = [0 0.1]
        [t,E,~] = trajectory(A2,m2,n2,1,1,eps,2e4,opts);
        fprintf('  m=%d n=%d eps=%-5g slope[1e3,2e4]=%+.3f\n', ...
                m2, n2, eps, loglogslope(t,E,1e3,2e4));
    end
end
fprintf('\n');
end

% ---------------------------------------------------------------- helpers
function A = spdmat(d)
B = randn(d);  A = B*B.'/d + 0.8*eye(d);
end

function [t,E,S] = trajectory(A,m,n,al,mu,eps,T,opts)
d  = m*n;
y0 = [0.4*randn(d,1); 0.3*randn(d,1); zeros(m,1); zeros(n,1)];
tq = unique(min([linspace(0,10,200), logspace(1,log10(T),4000)], T)).';
sol = ode89(@(t,y) rikfad_rhs(t,y,A,m,n,mu,al,eps), [0 T], y0, opts);
Y   = deval(sol, tq);
[E,S] = energy(Y,A,m,n);
t = tq;
end

function s = loglogslope(t,E,lo,hi)
k = t>lo & t<hi & E>0;
p = polyfit(log(t(k)), log(E(k)), 1);
s = p(1);
end
