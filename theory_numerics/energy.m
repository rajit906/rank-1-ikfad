function [E, S] = energy(Y, A, m, n)
%ENERGY  Mechanical energy E = f + |P|^2/2 with f = <X,AX>/2, and S = sum(R).
d = m*n;
X = Y(1:d, :);  P = Y(d+1:2*d, :);  R = Y(2*d+1:2*d+m, :);
E = 0.5*sum(X .* (A*X), 1).' + 0.5*sum(P.^2, 1).';
S = sum(R, 1).';
end
