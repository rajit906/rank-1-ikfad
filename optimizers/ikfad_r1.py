import torch
from torch.optim import Optimizer


class iKFAD_R1(Optimizer):
    """
    Rank-1 iKFAD (low-rank adaptive friction) - BACD splitting.

    Continuous intuition (matrix param X):
      Ẋ = P
      Ṗ = -∇f(X) - 1/2 * Xi ⊙ P - γ P
      ṙ = -α r + (1/(μ)) row_mean(P^2)
      ċ = -α c + (1/(μ)) col_mean(P^2)
      Xi = (r c^T) / Z

    Discrete implemented with BACD splitting (B, A, C, D).
    For non-2D params we fall back to elementwise ksi_buffer (same behavior
    as full iKFAD).
    """
    def __init__(self, params, h=1e-3, alpha=1.0, mu=1.0, gamma=0.0, eps_norm=1e-16):
        if h < 0:
            raise ValueError("Invalid step size h: should be >= 0.")
        if alpha < 0:
            raise ValueError("Invalid alpha: should be >= 0.")
        if mu <= 0:
            raise ValueError("Invalid mu: should be > 0.")
        if gamma < 0:
            raise ValueError("Invalid gamma: should be >= 0.")
        defaults = dict(h=float(h), alpha=float(alpha), mu=float(mu),
                        gamma=float(gamma), eps_norm=float(eps_norm))
        super().__init__(params, defaults)
        self.initialized = False

    @torch.no_grad()
    def initialize_state(self):
        """Create buffers: momentum_buffer and either (r,c) for 2D tensors or ksi for others."""
        for group in self.param_groups:
            for p in group['params']:
                if p is None:
                    continue
                state = self.state[p]
                shape = p.data.shape
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                if len(shape) == 2:
                    n, m = shape
                    if 'r_buffer' not in state:
                        state['r_buffer'] = torch.zeros(n, device=p.device, dtype=p.dtype)
                    if 'c_buffer' not in state:
                        state['c_buffer'] = torch.zeros(m, device=p.device, dtype=p.dtype)
                else:
                    if 'ksi_buffer' not in state:
                        state['ksi_buffer'] = torch.zeros_like(p.data)
        self.initialized = True

    # -------------------------
    # B step: p = p - h * grad
    # -------------------------
    @torch.no_grad()
    def B_step(self):
        for group in self.param_groups:
            h = group['h']
            for q in group['params']:
                if q.grad is None:
                    continue
                state = self.state[q]
                p_buf = state['momentum_buffer']
                g = q.grad
                # in-place momentum update
                p_buf.add_(g, alpha=-h)

    # -------------------------
    # A step: x = x + h * p
    # -------------------------
    @torch.no_grad()
    def A_step(self):
        for group in self.param_groups:
            h = group['h']
            for q in group['params']:
                state = self.state[q]
                p_buf = state['momentum_buffer']
                q.add_(p_buf, alpha=h)

    # -------------------------
    # C step: rank-1 friction half-damps + factor updates
    # -------------------------
    @torch.no_grad()
    def C_step(self):
        """
        Rank-1 friction update aligned with Adafactor paper logic.
        Uses sums instead of means and corrects normalization Z.
        """
        for group in self.param_groups:
            h = group['h']
            alpha = group['alpha']
            mu = group['mu']
            eps_norm = group.get('eps_norm', 1e-16)

            # Precompute exponential decay terms
            scalar_exp_alpha = torch.exp(torch.tensor(-alpha * h, dtype=torch.float32))

            for q in group['params']:
                if q.grad is None:
                    continue
                state = self.state[q]
                p_buf = state['momentum_buffer']
                shape = q.data.shape

                if len(shape) == 2:
                    r_buf = state['r_buffer']
                    c_buf = state['c_buffer']

                    # --- FIRST HALF-DAMP ---
                    # Reconstruction Z must be the total sum (1_n^T R)
                    Z = r_buf.sum().clamp_min(eps_norm)
                    # Xi = (r @ c.T) / sum(r)
                    Xi_old = torch.outer(r_buf, c_buf) / Z
                    p_buf.mul_(torch.exp(-0.5 * h * Xi_old))

                    # --- UPDATE FACTORS (Using Sums per Adafactor Eq 345) ---
                    p_sq = p_buf.pow(2)
                    row_sum = p_sq.sum(dim=1)   # 1_m (Eq 345)
                    col_sum = p_sq.sum(dim=0)   # 1_n^T (Eq 345)

                    s_exp = scalar_exp_alpha.to(device=r_buf.device, dtype=r_buf.dtype)
                    # Use (1-e)/alpha for continuous-to-discrete consistency
                    update_scale = (1.0 - s_exp) / (alpha * mu)
                    
                    r_buf.mul_(s_exp).add_(row_sum, alpha=update_scale)
                    c_buf.mul_(s_exp).add_(col_sum, alpha=update_scale)

                    # --- SECOND HALF-DAMP ---
                    Z_new = r_buf.sum().clamp_min(eps_norm)
                    Xi_new = torch.outer(r_buf, c_buf) / Z_new
                    p_buf.mul_(torch.exp(-0.5 * h * Xi_new))

                else:
                    # Fallback for vectors (Standard iKFAD behavior)
                    ksi_buf = state['ksi_buffer']
                    p_buf.mul_(torch.exp(-0.5 * h * ksi_buf))
                    
                    p_sq = p_buf.pow(2)
                    s_exp = scalar_exp_alpha.to(device=ksi_buf.device, dtype=ksi_buf.dtype)
                    ksi_buf.mul_(s_exp).add_(p_sq, alpha=(1.0 - s_exp) / (alpha * mu))
                    
                    p_buf.mul_(torch.exp(-0.5 * h * ksi_buf))

    # -------------------------
    # D step: global linear damping
    # -------------------------
    @torch.no_grad()
    def D_step(self):
        for group in self.param_groups:
            h = group['h']
            gamma = group['gamma']
            if gamma == 0.0:
                continue
            scalar = None
            for q in group['params']:
                if q.grad is None:
                    continue
                state = self.state[q]
                p_buf = state['momentum_buffer']
                if scalar is None:
                    scalar = torch.exp(torch.tensor(-gamma * h, device=p_buf.device, dtype=p_buf.dtype))
                p_buf.mul_(scalar)

    # -------------------------
    # Full BACD step
    # -------------------------
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single BACD integration step.
        If closure is provided, it will be called before B to (re)compute gradients.
        This matches the usual PyTorch pattern where you call loss.backward() pre-step,
        so closure is optional.
        """
        if not self.initialized:
            self.initialize_state()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.B_step()
        self.A_step()
        self.C_step()
        self.D_step()
        return loss
