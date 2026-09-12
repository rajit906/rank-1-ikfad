function dy = rikfad_rhs(~, y, A, m, n, mu, al, eps)
%RIKFAD_RHS  Right-hand side of the gamma=0 rank-1 iKFAD dynamics.
%   State y = [X(:); P(:); R; C],  X,P in R^{m x n},  R in R^m,  C in R^n.
d  = m*n;
X  = y(1:d);  P = y(d+1:2*d);
R  = y(2*d+1:2*d+m);  C = y(2*d+m+1:2*d+m+n);
Pm = reshape(P, m, n);
S  = sum(R);
den = S + eps;
if den > 0
    xi = (R * C.') / den;
else
    xi = zeros(m, n);
end
dX = P;
dP = -A*X - reshape(xi .* Pm, d, 1);
dR = sum(Pm.^2, 2)/mu - al*R;
dC = (sum(Pm.^2, 1).')/mu - al*C;
dy = [dX; dP; dR; dC];
end
