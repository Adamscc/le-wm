import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift


def _t_dist_cf_numpy(t_vals: np.ndarray, nu: float) -> np.ndarray:
    """Compute the characteristic function of a standard t(nu) distribution.

    For a symmetric distribution the CF is real-valued:
        φ(t; ν) = K_{ν/2}(√ν |t|) * (√ν |t|)^{ν/2} / (2^{ν/2 - 1} * Γ(ν/2))
    with φ(0; ν) = 1 by continuity.

    Uses scipy, so only call this at init time (not in the training loop).
    """
    from scipy.special import kv, gamma as sp_gamma

    t_abs = np.abs(t_vals).astype(np.float64)
    phi = np.ones_like(t_abs)

    mask = t_abs > 1e-8
    ta = t_abs[mask]

    order = nu / 2.0
    arg = np.sqrt(nu) * ta

    bessel = kv(order, arg)
    numerator = bessel * arg ** order
    denominator = 2.0 ** (order - 1.0) * sp_gamma(order)
    phi[mask] = numerator / denominator

    # Clip to [0, 1]: numerical noise can push values slightly outside
    return np.clip(phi, 0.0, 1.0).astype(np.float32)


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!).

    When rho=0 (default), target is N(0, I) via random projections — original behaviour.
    When rho>0, implements Whitened Adaptive SIGReg:
        z ~ N(0, diag(sigma^2))  iff  z / sigma ~ N(0, I)
    The latent is whitened by the learnable per-dim std sigma, then the original
    random-projection SIGReg is applied to the whitened latent.  This checks the
    full joint distribution (not just per-axis marginals) while still allowing an
    anisotropic target covariance.
    Parameterisation: sigma_i^2 = (1-rho) + rho*D*softmax(alpha)_i,
    which keeps mean(sigma^2) == 1 and each sigma_i^2 >= (1-rho).
    """

    def __init__(self, knots=17, num_proj=1024, dim=None, rho=0.0, beta_sigma=1e-3):
        super().__init__()
        self.num_proj = num_proj
        self.rho = rho
        self.beta_sigma = beta_sigma

        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

        if rho > 0.0:
            assert dim is not None, "dim must be provided when rho > 0"
            self.dim = dim
            # init to zeros so softmax is uniform => sigma^2 = 1, same as isotropic
            self.alpha = nn.Parameter(torch.zeros(dim))

    def _get_sigma2(self):
        """Per-dimension target variance; mean == 1, each dim >= (1 - rho)."""
        p = torch.softmax(self.alpha, dim=0)
        return (1.0 - self.rho) + self.rho * self.dim * p

    def _random_sigreg(self, z):
        """Standard random-projection SIGReg against N(0, I). z: (T, B, D)."""
        A = torch.randn(z.size(-1), self.num_proj, device=z.device, dtype=z.dtype)
        A = A.div_(A.norm(p=2, dim=0))

        x_t = (z @ A).unsqueeze(-1) * self.t
        cos_emp = x_t.cos().mean(-3)  # mean over B: (T, M, K)
        sin_emp = x_t.sin().mean(-3)  # (T, M, K)

        err = (cos_emp - self.phi).pow(2) + sin_emp.pow(2)
        statistic = (err @ self.weights) * z.size(-2)
        return statistic.mean()

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        if self.rho > 0.0:
            # Whiten: z_white = z / sigma  =>  z_white ~ N(0, I)  if  z ~ N(0, diag(sigma^2))
            sigma2 = self._get_sigma2().to(proj.dtype)          # (D,)
            sigma = torch.sqrt(sigma2 + 1e-6)                   # (D,)
            proj_white = proj / sigma.view(1, 1, -1)            # (T, B, D)

            loss = self._random_sigreg(proj_white)
            loss = loss + self.beta_sigma * self.alpha.pow(2).mean()
        else:
            loss = self._random_sigreg(proj)

        return loss


# =====================================================================
#  t-Distribution SIGReg
# =====================================================================

class TDistSIGReg(nn.Module):
    """SIGReg with a Student-t target distribution.

    Replaces the Gaussian characteristic function φ_0(t) = exp(-t²/2) with
    the t-distribution CF computed via modified Bessel functions (scipy).
    The CF is precomputed once at init and stored as a fixed buffer — no
    gradient flows through ν.

    For ν → ∞ this degenerates to the standard Gaussian SIGReg.
    For small ν (e.g. 3–5) it encourages heavier-tailed latents, which can
    reduce PR while preserving SR.

    Args:
        nu:       Degrees of freedom (float, must be > 2 for finite variance).
        knots:    Number of quadrature points in [0, t_max].
        t_max:    Upper limit of the integration interval (default 4.0).
                  t-dist CFs decay slower than Gaussian, so 4.0 > 3.0 default.
        num_proj: Number of random projection directions (Monte-Carlo average).
    """

    def __init__(self, nu: float = 5.0, knots: int = 17, t_max: float = 4.0, num_proj: int = 1024, **kwargs):
        super().__init__()
        assert nu > 2.0, "nu must be > 2 for a finite-variance t-distribution"
        self.num_proj = num_proj
        self.nu = nu
        self.rho = 0.0  # compatibility attribute checked in train.py

        t_np = np.linspace(0.0, t_max, knots, dtype=np.float32)
        phi_np = _t_dist_cf_numpy(t_np, nu)

        dt = t_max / max(knots - 1, 1)
        weights_np = np.full(knots, 2.0 * dt, dtype=np.float32)
        weights_np[0] = dt
        weights_np[-1] = dt
        window_np = np.exp(-t_np ** 2 / 2.0)  # same Gaussian envelope as SIGReg

        self.register_buffer("t", torch.from_numpy(t_np))
        self.register_buffer("phi", torch.from_numpy(phi_np))
        self.register_buffer("weights", torch.from_numpy(weights_np * window_np))

    def _random_sigreg(self, z: torch.Tensor) -> torch.Tensor:
        """Standard random-projection SIGReg against the precomputed t-dist CF.

        z: (T, B, D)
        """
        A = torch.randn(z.size(-1), self.num_proj, device=z.device, dtype=z.dtype)
        A = A.div_(A.norm(p=2, dim=0))

        x_t = (z @ A).unsqueeze(-1) * self.t       # (T, B, M, K)
        cos_emp = x_t.cos().mean(-3)                # (T, M, K)
        sin_emp = x_t.sin().mean(-3)                # (T, M, K)

        # t-dist is symmetric → CF is real; imaginary target is 0
        err = (cos_emp - self.phi).pow(2) + sin_emp.pow(2)
        statistic = (err @ self.weights) * z.size(-2)
        return statistic.mean()

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """proj: (T, B, D)"""
        return self._random_sigreg(proj)


# =====================================================================
#  Route 1 — Effective Rank Regularization
# =====================================================================

def effective_rank_loss(z, min_var_threshold=0.01):
    """Negative normalised effective rank + per-dim minimum variance penalty.

    Maximises the entropy of the normalised eigenvalue distribution of the
    batch covariance, which prevents dimensional collapse while remaining
    agnostic to any specific target spectrum.

    Args:
        z: (T, B, D) latent embeddings (same layout as SIGReg input).
        min_var_threshold: floor on per-dimension variance; set 0 to disable.

    Returns:
        Scalar in roughly [-1, 0] (rank term) plus a non-negative penalty.
    """
    z_flat = z.reshape(-1, z.size(-1)).float()
    D = z_flat.size(-1)
    z_c = z_flat - z_flat.mean(dim=0, keepdim=True)
    cov = z_c.T @ z_c / (z_flat.size(0) - 1)

    eigvals = torch.linalg.eigvalsh(cov).clamp(min=1e-8)
    p = eigvals / eigvals.sum()
    eff_rank = torch.exp(-(p * p.log()).sum())
    loss = -eff_rank / D                        # in [-1, 0]

    if min_var_threshold > 0:
        per_dim_var = torch.diagonal(cov)
        loss = loss + F.relu(min_var_threshold - per_dim_var).mean()

    return loss


# =====================================================================
#  Route 2 — Soft-Whitening SIGReg
# =====================================================================

class SoftWhiteningSIGReg(nn.Module):
    """Blend raw latent with ZCA-whitened latent, then apply random-proj SIGReg.

    z_blended = (1 - tau) * z + tau * ZCA_whiten(z)

    tau is a sigmoid-parameterised learnable scalar.
        tau -> 0  :  original SIGReg on the raw representation
        tau -> 1  :  full whitening (SIGReg trivially zero)
    The equilibrium tau balances decorrelation vs. preserving useful structure.
    """

    def __init__(self, knots=17, num_proj=1024, eps=1e-5, **kwargs):
        super().__init__()
        self.num_proj = num_proj
        self.eps = eps
        self.rho = 0.0                                    # compat with diag logging
        self.tau_logit = nn.Parameter(torch.zeros(1))      # sigmoid(0) = 0.5

        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    @property
    def tau(self):
        return torch.sigmoid(self.tau_logit)

    def _random_sigreg(self, z):
        A = torch.randn(z.size(-1), self.num_proj, device=z.device, dtype=z.dtype)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (z @ A).unsqueeze(-1) * self.t
        cos_emp = x_t.cos().mean(-3)
        sin_emp = x_t.sin().mean(-3)
        err = (cos_emp - self.phi).pow(2) + sin_emp.pow(2)
        statistic = (err @ self.weights) * z.size(-2)
        return statistic.mean()

    def forward(self, proj):
        """proj: (T, B, D)"""
        T, B, D = proj.shape
        tau = self.tau

        z_flat = proj.reshape(-1, D).float()
        z_c = z_flat - z_flat.mean(dim=0, keepdim=True)
        cov = z_c.T @ z_c / (z_flat.size(0) - 1)

        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = eigvals.clamp(min=self.eps)
        W_zca = eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.T
        z_white = (z_c @ W_zca).to(proj.dtype).reshape(T, B, D)

        z_blended = (1.0 - tau) * proj + tau * z_white
        return self._random_sigreg(z_blended)


# =====================================================================
#  Route 3 — Spectral Regularizer (learnable exponential-decay target)
# =====================================================================

class SpectralRegularizer(nn.Module):
    """Constrain the covariance eigenspectrum toward a learnable exponential decay.

    Target eigenvalues:  lambda_i ~ exp(-decay_rate * i)
    re-scaled to match the actual total variance.

      decay_rate -> 0  :  isotropic (all eigenvalues equal, like original SIGReg)
      decay_rate > 0   :  controlled anisotropy (front PCs dominate)

    A quadratic penalty on decay_rate biases toward isotropy unless the data
    strongly demands a skewed spectrum.  Compose with SIGReg (rho=0) for full
    regularisation: SIGReg controls overall Gaussianity; this shapes the spectrum.
    """

    def __init__(self, dim, init_decay=0.02, max_decay=5.0, decay_penalty=0.01):
        super().__init__()
        self.dim = dim
        self.max_decay = max_decay
        self.decay_penalty = decay_penalty
        init_logit = math.log(init_decay / max(max_decay - init_decay, 1e-6))
        self.decay_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        self.register_buffer("indices", torch.arange(dim, dtype=torch.float32))

    @property
    def decay_rate(self):
        return torch.sigmoid(self.decay_logit) * self.max_decay

    def forward(self, z):
        """z: (T, B, D)"""
        z_flat = z.reshape(-1, z.size(-1)).float()
        z_c = z_flat - z_flat.mean(dim=0, keepdim=True)
        cov = z_c.T @ z_c / (z_flat.size(0) - 1)

        eigvals = torch.linalg.eigvalsh(cov).clamp(min=0)
        eigvals_desc = eigvals.flip(0)

        decay = self.decay_rate
        target = torch.exp(-decay * self.indices)
        target = target / target.sum() * eigvals_desc.sum().detach()

        spectral_loss = F.mse_loss(eigvals_desc, target)
        return spectral_loss + self.decay_penalty * decay.pow(2)


class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x
