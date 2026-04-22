import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import h5py
import numpy as np
from einops import rearrange
from pathlib import Path

from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm
import hdf5plugin
import csv
import json
from datetime import datetime
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


def scaled_dot_product_attention(Q, K, V, is_causal=True, is_traning=False, dropout=0.0):
    '''
    Q (batch, seq_len, d_k)
    K (batch, seq_len, d_k)
    V (batch, seq_len, d_k)
    '''

    d_k = Q.size(-1)
    scores = Q @ K.transpose(-1, -2)
    scores = scores / math.sqrt(d_k)
    if is_causal:
        ones = torch.ones(scores.size(-2), scores.size(-1), device=scores.device)
        mask = torch.tril(ones)
        scores = scores.masked_fill(mask == 0, value=-1e9)
    weights = F.softmax(scores, dim=-1)
    weights = F.dropout(weights, p=dropout, training=is_traning)
    output = weights @ V
    return output, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, head, head_dim, dim, dropout=0.0):
        super().__init__()
        self.h = head
        self.dim = dim
        self.head_dim = head_dim
        self.head = head
        self.linear_q = nn.Linear(in_features=dim, out_features=head*head_dim, bias=False)
        self.linear_k = nn.Linear(in_features=dim, out_features=head*head_dim, bias=False)
        self.linear_v = nn.Linear(in_features=dim, out_features=head*head_dim, bias=False)
        self.dropout = dropout
        self.out = nn.Sequential(
            nn.Linear(in_features=head*head_dim, out_features=dim),
            nn.Dropout(dropout)
        )

    def forward(self, Q, K, V, is_causal=True):
        batch = Q.size(0)
        seq_len = Q.size(1)
        Q = self.linear_q(Q) #(B T H*H_D)
        K = self.linear_k(K)
        V = self.linear_v(V)

        Q = Q.view(batch, seq_len, self.h, self.head_dim).transpose(1, 2) #(B H T H_D)
        K = K.view(batch, seq_len, self.h, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.h, self.head_dim).transpose(1, 2)

        output, weights = scaled_dot_product_attention(Q, K, V, is_causal=is_causal, is_traning=self.training,
                                                       dropout=self.dropout)
        # output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.head*self.head_dim)
        output = rearrange(output, 'b h t h_d -> b t (h h_d)')
        output = self.out(output)

        return output, weights


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.out(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, mlp_dim, head, head_dim, dropout=0.0, is_causal=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.attention = MultiHeadAttention(dim=dim, head_dim=head_dim, head=head, dropout=dropout)
        self.ffn = FFN(dim=dim, hidden_dim=mlp_dim, dropout=dropout)
        self.is_causal = is_causal

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, _ = self.attention(norm_x, norm_x, norm_x, is_causal=self.is_causal)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class ConditionalTransformerBlock(nn.Module):
    def __init__(self, dim, mlp_dim, head, head_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attention = MultiHeadAttention(dim=dim, head_dim=head_dim, head=head, dropout=dropout)
        self.ffn = FFN(dim=dim, hidden_dim=mlp_dim, dropout=dropout)
        self.modulate = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=dim, out_features=dim * 6)
        )
        nn.init.constant_(self.modulate[-1].weight, 0)
        nn.init.constant_(self.modulate[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = self.modulate(c).chunk(6, dim=-1)
        norm1_x = self.norm1(x)
        modulate1_x = norm1_x * (1 + scale_msa) + shift_msa
        attn_out, _ = self.attention(modulate1_x, modulate1_x, modulate1_x)
        x = x + attn_out * gate_msa
        norm2_x = self.norm2(x)
        modulate2_x = norm2_x * (1 + scale_ffn) + shift_ffn
        x = x + self.ffn(modulate2_x) * gate_ffn
        return x


class Predictor(nn.Module):
    def __init__(self, num_frame, in_dim, hidden_dim, out_dim,
                 mlp_dim, head, head_dim, depth, dropout=0.0, emb_dropout=0.0):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, num_frame, in_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.in_proj = (
            nn.Linear(in_features=in_dim, out_features=hidden_dim)
            if in_dim != hidden_dim else nn.Identity())
        self.con_proj = (
            nn.Linear(in_features=in_dim, out_features=hidden_dim)
            if in_dim != hidden_dim else nn.Identity())
        self.out_proj = (
            nn.Linear(in_features=hidden_dim, out_features=out_dim)
            if out_dim != hidden_dim else nn.Identity())
        self.transformer = nn.ModuleList(
            [ConditionalTransformerBlock(dim=hidden_dim, mlp_dim=mlp_dim,
                                         head=head, head_dim=head_dim, dropout=dropout)
             for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, c=None):
        T = x.size(1)
        x = x + self.pos_emb[:, :T]
        x = self.dropout(x)
        x = self.in_proj(x)
        if c is not None:
            c = self.con_proj(c)
        for layer in self.transformer:
            x = layer(x, c)
        x = self.norm(x)
        x = self.out_proj(x)
        return x


class SIGReg(nn.Module):
    def __init__(self, num_proj, num_knot):
        super().__init__()
        self.num_proj = num_proj #采样的线的数量，即投影检验的次数
        self.num_knot = num_knot #梯形积分法的采样点的数量
        t = torch.linspace(0, 3, num_knot) #梯形积分法采样点，因为在3之后权重函数就已经非常接近0，所以积分范围取0-3
        phi = torch.exp(-t**2/2) #公式里的一个常数
        dt = 3 / (num_knot - 1) #积分总长度3，分成knot段
        weights = torch.full((num_knot, ), 2 * dt)
        weights[[0, -1]] = dt
        weights = weights * phi
        self.register_buffer('t', t)
        self.register_buffer('phi', phi)
        self.register_buffer('weights', weights)

    def forward(self, Z):
        A = torch.randn(Z.size(-1), self.num_proj, device=Z.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (Z @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * Z.size(-2)
        return statistic.mean()

class Encoder(nn.Module):
    def __init__(self, hidden_dim, patch_size, depth, head, img_size, dropout=0.0):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_emb = nn.Parameter(torch.randn(1, (img_size[0]//patch_size)**2+1, hidden_dim))
        self.transformer = nn.ModuleList([TransformerBlock(dim=hidden_dim, mlp_dim=hidden_dim*4,
                                                           head_dim=hidden_dim//head, head=head,
                                                           is_causal=False, dropout=dropout)
                                          for _ in range(depth)])
        self.cls = nn.Parameter(torch.randn(1, hidden_dim))

    def forward(self, x):
        x = self.init_conv(x)
        x_rearange = rearrange(x, 'b d h w -> b (h w) d')
        x = torch.cat([self.cls.unsqueeze(0).expand(x_rearange.size(0), -1, -1), x_rearange], dim=1)
        x = x + self.pos_emb
        for layer in self.transformer:
            x = layer(x)
        return x[:, 0, :]

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(

            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class Action_encoder(nn.Module):
    def __init__(self, in_dim=10, hidden_dim=10, out_dim=10):
        super().__init__()
        self.proj = nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=4*out_dim),
            nn.SiLU(),
            nn.Linear(in_features=4*out_dim, out_features=out_dim)
        )

    def forward(self, x):
        x = x.float()
        x = self.proj(x)
        x = self.mlp(x)
        return x


class Lewm(nn.Module):
    def __init__(self, encoder, predictor, action_encoder, projector=None, pred_projector=None):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector if projector is not None else nn.Identity()
        self.pred_projector = pred_projector if pred_projector is not None else nn.Identity()

    def encode(self, x):
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        emb = self.encoder(x)
        emb = self.projector(emb) # (b t) d
        return emb

    def predict(self, emb, action_emb=None):
        pred = self.predictor(emb, action_emb)
        pred = rearrange(pred, 'b t d -> (b t) d')
        pred = self.pred_projector(pred)
        pred = rearrange(pred, '(b t) d -> b t d', b=emb.size(0))
        return pred

    def rollout(self, info, action_seq, history_size=3):
        B, S, T = action_seq.shape[:3]
        H = info['pixels'].size(2)
        act_init, act_future = torch.split(action_seq, [H, T - H], dim=2)
        info['action'] = act_init
        n_steps = T - H
        emb_init = info['pixels'][:, 0]
        emb_init = self.encode(emb_init)
        emb_init = rearrange(emb_init, '(b t) d -> b t d', b=B)
        emb_init = emb_init.unsequeeze(1).expand(B, S, -1, -1)
        info['emb'] = emb_init
        emb = rearrange(emb_init, 'b s t d -> (b s) t d')
        act = rearrange(act_init, 'b s t d -> (b s) t d')
        act_future = rearrange(act_future, 'b s t d -> (b s) t d')

        HS = history_size
        for t in n_steps:
            emb_hs = emb[:, -HS:]
            act_emb = self.action_encoder(act)[:, -HS:]
            next_emb = self.predict(emb_hs, act_emb)[:, -1:, :] #(b s) 1 d
            emb = torch.cat([emb, next_emb], dim=1)#(b s) t+1 d
            next_act = act_future[:, t:t+1, :]
            act = torch.cat([act, next_act], dim=0)

        emb_hs = emb[:, -HS:]
        act_emb = self.action_encoder(act)[:, -HS:]
        next_emb = self.predict(emb_hs, act_emb)[:, -1:, :]  # (b s) 1 d
        emb = torch.cat([emb, next_emb], dim=1)

        res_emb =rearrange(emb, '(b s) t d -> b s t d', b=B)
        info['pred_emb'] = res_emb
        return info

    def criterion(self, info):
        pass


    def get_cost(self, info):
        pass













class TwoRoomH5Dataset(Dataset):
    """
    Build sequence samples from tworoom.h5.

    Output:
      - pixels: (T, C, H, W), float32 in [0, 1]
      - action: (T, frameskip * action_dim), float32
    """

    def __init__(self, h5_path, num_steps, frameskip, max_samples=None):
        self.h5_path = str(h5_path)
        self.num_steps = num_steps
        self.frameskip = frameskip
        self.max_samples = max_samples

        with h5py.File(self.h5_path, "r") as f:
            self.ep_offset = np.asarray(f["ep_offset"][:], dtype=np.int64)
            self.ep_len = np.asarray(f["ep_len"][:], dtype=np.int64)
            self.action_dim = int(f["action"].shape[-1])
            action_data = np.asarray(f["action"][:], dtype=np.float32)
            finite_mask = np.isfinite(action_data).all(axis=1)
            action_data = action_data[finite_mask]
            action_mean = action_data.mean(axis=0)
            action_std = action_data.std(axis=0)
            action_std = np.where(action_std < 1e-6, 1.0, action_std)
            self.action_mean = torch.from_numpy(action_mean).float()
            self.action_std = torch.from_numpy(action_std).float()

        # Need T*frameskip actions to pack per model step.
        required_len = self.num_steps * self.frameskip
        starts = []
        for start, length in zip(self.ep_offset, self.ep_len):
            if length < required_len:
                continue
            max_start = int(start + length - required_len)
            starts.extend(range(int(start), max_start + 1))

        if self.max_samples is not None:
            starts = starts[: self.max_samples]
        self.starts = np.asarray(starts, dtype=np.int64)

        self._h5 = None

    def _get_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        f = self._get_h5()
        start = int(self.starts[idx])

        # Model-time observation indices.
        pixel_idx = start + np.arange(self.num_steps, dtype=np.int64) * self.frameskip
        pixels = f["pixels"][pixel_idx]  # (T, H, W, C), uint8
        pixels = torch.from_numpy(pixels).float().permute(0, 3, 1, 2) / 255.0
        # Match train.py image preprocessor behavior (ImageNet normalization).
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=pixels.dtype).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=pixels.dtype).view(1, 3, 1, 1)
        pixels = (pixels - imagenet_mean) / imagenet_std

        # Pack frameskip raw actions into one model-time action token.
        act = f["action"][start : start + self.num_steps * self.frameskip]  # (T*F, A)
        act = torch.from_numpy(np.asarray(act, dtype=np.float32))
        act = torch.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0)
        act = (act - self.action_mean) / self.action_std
        act = act.view(self.num_steps, self.frameskip, self.action_dim)
        act = act.reshape(self.num_steps, self.frameskip * self.action_dim)

        return {"pixels": pixels, "action": act}


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def run_random_test(model, sigreg, batch_size, time_steps, history_size, num_pred, action_dim, frameskip, img_size):
    device = next(model.parameters()).device
    model.train()

    # Random synthetic batch that matches the expected training input layout.
    pixels = torch.randn(batch_size, time_steps, 3, img_size[0], img_size[1], device=device)
    actions = torch.randn(batch_size, time_steps, action_dim * frameskip, device=device)

    emb = model.encode(pixels)  # (B*T, D)
    emb = emb.view(batch_size, time_steps, -1)  # (B, T, D)
    act_emb = model.action_encoder(actions)  # (B, T, D)

    ctx_emb = emb[:, :history_size]
    ctx_act = act_emb[:, :history_size]
    tgt_emb = emb[:, num_pred:]
    pred_emb = model.predict(ctx_emb, ctx_act)

    pred_loss = (pred_emb - tgt_emb).pow(2).mean()
    reg_loss = sigreg(emb.transpose(0, 1))
    total_loss = pred_loss + 0.09 * reg_loss

    print("=== Random Forward Test ===")
    print(f"pixels shape:   {tuple(pixels.shape)}")
    print(f"actions shape:  {tuple(actions.shape)}")
    print(f"emb shape:      {tuple(emb.shape)}")
    print(f"act_emb shape:  {tuple(act_emb.shape)}")
    print(f"pred_emb shape: {tuple(pred_emb.shape)}")
    print(f"tgt_emb shape:  {tuple(tgt_emb.shape)}")
    print(f"pred_loss:      {pred_loss.item():.6f}")
    print(f"sigreg_loss:    {reg_loss.item():.6f}")
    print(f"total_loss:     {total_loss.item():.6f}")


@torch.no_grad()
def collapse_diagnostics(pred_emb, tgt_emb):
    """
    Collapse checks for embeddings of shape (B, T, D).
    """
    eps = 1e-8

    def mean_offdiag_cos(x):
        # x: (N, D)
        n = x.size(0)
        if n < 2:
            return 0.0
        x = F.normalize(x, dim=-1, eps=eps)
        sim = x @ x.transpose(0, 1)
        offdiag_sum = sim.sum() - sim.diag().sum()
        return (offdiag_sum / (n * (n - 1))).item()

    pred_var = pred_emb.var(dim=(0, 1), unbiased=False).mean().item()
    tgt_var = tgt_emb.var(dim=(0, 1), unbiased=False).mean().item()

    pred_flat = pred_emb.reshape(-1, pred_emb.size(-1))
    tgt_flat = tgt_emb.reshape(-1, tgt_emb.size(-1))
    pred_offdiag_cos = mean_offdiag_cos(pred_flat)
    tgt_offdiag_cos = mean_offdiag_cos(tgt_flat)

    if pred_emb.size(1) > 1:
        pred_temporal_delta = (pred_emb[:, 1:] - pred_emb[:, :-1]).pow(2).mean().item()
        tgt_temporal_delta = (tgt_emb[:, 1:] - tgt_emb[:, :-1]).pow(2).mean().item()
    else:
        pred_temporal_delta = 0.0
        tgt_temporal_delta = 0.0

    suspicious = (
        pred_var < 1e-4
        and pred_temporal_delta < 1e-4
        and pred_offdiag_cos > 0.98
    )

    return {
        "pred_var": pred_var,
        "tgt_var": tgt_var,
        "pred_offdiag_cos": pred_offdiag_cos,
        "tgt_offdiag_cos": tgt_offdiag_cos,
        "pred_temporal_delta": pred_temporal_delta,
        "tgt_temporal_delta": tgt_temporal_delta,
        "suspicious": suspicious,
    }


def plot_training_curves(metrics_path, output_dir):
    rows = []
    with open(metrics_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return

    steps = [int(r["global_step"]) for r in rows]
    loss = [float(r["loss"]) for r in rows]
    pred_loss = [float(r["pred_loss"]) for r in rows]
    reg_loss = [float(r["reg_loss"]) for r in rows]

    pred_var = [float(r["pred_var"]) for r in rows if r["pred_var"] != "nan"]
    tgt_var = [float(r["tgt_var"]) for r in rows if r["tgt_var"] != "nan"]
    pred_dt = [float(r["pred_temporal_delta"]) for r in rows if r["pred_temporal_delta"] != "nan"]
    tgt_dt = [float(r["tgt_temporal_delta"]) for r in rows if r["tgt_temporal_delta"] != "nan"]
    pred_cos = [float(r["pred_offdiag_cos"]) for r in rows if r["pred_offdiag_cos"] != "nan"]
    tgt_cos = [float(r["tgt_offdiag_cos"]) for r in rows if r["tgt_offdiag_cos"] != "nan"]
    diag_steps = [int(r["global_step"]) for r in rows if r["pred_var"] != "nan"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(steps, loss, label="loss")
    axes[0, 0].plot(steps, pred_loss, label="pred_loss")
    axes[0, 0].plot(steps, reg_loss, label="reg_loss")
    axes[0, 0].set_title("Training Losses")
    axes[0, 0].set_xlabel("global_step")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(diag_steps, pred_var, label="pred_var")
    axes[0, 1].plot(diag_steps, tgt_var, label="tgt_var")
    axes[0, 1].set_title("Embedding Variance")
    axes[0, 1].set_xlabel("global_step")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(diag_steps, pred_dt, label="pred_dt")
    axes[1, 0].plot(diag_steps, tgt_dt, label="tgt_dt")
    axes[1, 0].set_title("Temporal Delta")
    axes[1, 0].set_xlabel("global_step")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(diag_steps, pred_cos, label="pred_cos")
    axes[1, 1].plot(diag_steps, tgt_cos, label="tgt_cos")
    axes[1, 1].set_title("Mean Off-diagonal Cosine")
    axes[1, 1].set_xlabel("global_step")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(Path(output_dir) / "training_curves.png", dpi=180)
    plt.close(fig)


def train_with_adamw(
    model,
    sigreg,
    train_loader,
    history_size,
    num_pred,
    lambda_sigreg=0.09,
    epochs=10,
    lr=5e-5,
    weight_decay=1e-3,
    diag_every=20,
    log_dir=None,
    warmup_ratio=0.1,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model = model.to(device)
    sigreg = sigreg.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    warmup_steps = max(1, int(total_steps * warmup_ratio))  # 确保至少有 1 步 warmup

    # 1. 预热阶段：学习率从 lr * 1e-3 线性增长到 lr
    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
    )
    # 2. 退火阶段：学习率从 lr 按照余弦曲线衰减到接近 0 (1e-6)
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
    )
    # 3. 拼接调度器：在 warmup_steps 时点进行切换
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )
    log_dir = Path(log_dir or (Path("outputs") / "debug_train"))
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / "metrics.csv"
    summary_path = log_dir / "summary.json"

    metric_fields = [
        "epoch",
        "step_in_epoch",
        "global_step",
        "loss",
        "pred_loss",
        "reg_loss",
        "pred_var",
        "tgt_var",
        "pred_temporal_delta",
        "tgt_temporal_delta",
        "pred_offdiag_cos",
        "tgt_offdiag_cos",
        "suspicious",
    ]
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metric_fields)
        writer.writeheader()

    checkpoints_dir = log_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoints_dir / "best.pt"
    last_ckpt_path = checkpoints_dir / "last.pt"
    best_loss = float("inf")

    global_step = 0
    epoch_avg_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)
        for step, batch in enumerate(pbar, start=1):
            global_step += 1
            pixels = batch["pixels"].to(device)    # (B, T, C, H, W)
            actions = batch["action"].to(device)   # (B, T, A_eff)
            actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)

            emb = model.encode(pixels).view(pixels.size(0), pixels.size(1), -1)  # (B, T, D)
            act_emb = model.action_encoder(actions)  # (B, T, D)

            ctx_emb = emb[:, :history_size]
            ctx_act = act_emb[:, :history_size]
            tgt_emb = emb[:, num_pred:]
            pred_emb = model.predict(ctx_emb, ctx_act)

            pred_loss = (pred_emb - tgt_emb).pow(2).mean()
            reg_loss = sigreg(emb.transpose(0, 1))
            loss = pred_loss + lambda_sigreg * reg_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                pred=f"{pred_loss.item():.4f}",
                reg=f"{reg_loss.item():.4f}",
                lr=f"{current_lr:.2e}",
            )
            diag = {
                "pred_var": float("nan"),
                "tgt_var": float("nan"),
                "pred_temporal_delta": float("nan"),
                "tgt_temporal_delta": float("nan"),
                "pred_offdiag_cos": float("nan"),
                "tgt_offdiag_cos": float("nan"),
                "suspicious": False,
            }
            if step % diag_every == 0:
                diag = collapse_diagnostics(pred_emb.detach(), tgt_emb.detach())
            with open(metrics_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=metric_fields)
                writer.writerow(
                    {
                        "epoch": epoch + 1,
                        "step_in_epoch": step,
                        "global_step": global_step,
                        "loss": loss.item(),
                        "pred_loss": pred_loss.item(),
                        "reg_loss": reg_loss.item(),
                        "pred_var": diag["pred_var"],
                        "tgt_var": diag["tgt_var"],
                        "pred_temporal_delta": diag["pred_temporal_delta"],
                        "tgt_temporal_delta": diag["tgt_temporal_delta"],
                        "pred_offdiag_cos": diag["pred_offdiag_cos"],
                        "tgt_offdiag_cos": diag["tgt_offdiag_cos"],
                        "suspicious": int(bool(diag["suspicious"])),
                    }
                )

        avg = running_loss / max(len(train_loader), 1)
        epoch_avg_losses.append(avg)

        # Save epoch checkpoint
        epoch_ckpt_path = checkpoints_dir / f"epoch_{epoch + 1:03d}.pt"
        ckpt_payload = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "avg_loss": avg,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        # torch.save(ckpt_payload, epoch_ckpt_path)

        # Always update last checkpoint
        torch.save(ckpt_payload, last_ckpt_path)

        # Update best checkpoint
        if avg < best_loss:
            best_loss = avg
            torch.save(ckpt_payload, best_ckpt_path)

    plot_training_curves(metrics_path, log_dir)
    summary = {
        "device": str(device),
        "epochs": epochs,
        "batches_per_epoch": len(train_loader),
        "epoch_avg_losses": epoch_avg_losses,
        "metrics_csv": str(metrics_path),
        "curves_png": str(log_dir / "training_curves.png"),
        "checkpoints_dir": str(checkpoints_dir),
        "last_ckpt": str(last_ckpt_path),
        "best_ckpt": str(best_ckpt_path),
        "best_avg_loss": best_loss,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    print("===========================begin================================")
    #main config
    history_size = 1
    num_pred = 1
    frameskip = 5
    lambda_sigreg = 0.09
    train_split = 0.9
    seed = 3072
    time_steps = history_size + num_pred

    #encoder config
    emb_dim = 192
    patch_size = 14
    depth = 12
    head = 3
    img_size = (224, 224)

    # 实例化
    encoder = Encoder(
        hidden_dim=emb_dim,
        patch_size=patch_size,
        depth=depth,
        head=head,
        img_size=img_size,
    )

    #action encoder config
    action_dim = 2
    action_encoder = Action_encoder(in_dim=action_dim*frameskip, hidden_dim=action_dim*frameskip, out_dim=emb_dim)

    #projector config
    projector = MLP(in_dim=emb_dim, hidden_dim=2048, out_dim=emb_dim)

    #pred_projector
    pred_projector = MLP(in_dim=emb_dim, hidden_dim=2048, out_dim=emb_dim)

    #predictor config
    mlp_dim = 2048
    head_pred = 16
    depth_pred = 6
    dim_head_pred = 64
    dropout = 0.1
    predictor = Predictor(in_dim=emb_dim, hidden_dim=emb_dim, out_dim=emb_dim, mlp_dim=mlp_dim, head_dim=dim_head_pred,
                          head=head_pred, depth=depth_pred, dropout=dropout, num_frame=history_size)

    #sigreg config
    num_proj = 1024
    num_knots = 17
    sigreg = SIGReg(num_proj=num_proj, num_knot=num_knots)

    lewm = Lewm(encoder=encoder, action_encoder=action_encoder, projector=projector, predictor=predictor, pred_projector=pred_projector)
    total_params, trainable_params = count_parameters(lewm)

    # # # Random data sanity check
    # batch_size = 2
    #
    # run_random_test(
    #     model=lewm,
    #     sigreg=sigreg,
    #     batch_size=batch_size,
    #     time_steps=time_steps,
    #     history_size=history_size,
    #     num_pred=num_pred,
    #     action_dim=action_dim,
    #     frameskip=frameskip,
    #     img_size=img_size,
    # )

    # HDF5 dataset + AdamW training
    h5_path = "/kaggle/input/datasets/adamscc/towroom-full/tworoom.h5"
    dataset = TwoRoomH5Dataset(
        h5_path=h5_path,
        num_steps=time_steps,
        frameskip=frameskip,
    )
    rnd_gen = torch.Generator().manual_seed(seed)
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=rnd_gen)

    loader = DataLoader(
        train_set,
        batch_size=768,
        shuffle=True,
        num_workers=12,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=True
    )

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "/kaggle/working/outputstrain_logs_lewm09"
    summary = train_with_adamw(
        model=lewm,
        sigreg=sigreg,
        train_loader=loader,
        history_size=history_size,
        num_pred=num_pred,
        lambda_sigreg=lambda_sigreg,
        epochs=10,
        lr=5e-5,
        weight_decay=1e-3,
        log_dir=log_dir,
        warmup_ratio=0.1
    )
    with open(f"{log_dir}/run_info.json", "w") as f:
        json.dump(
            {
                "h5_path": str(h5_path),
                "num_samples": len(dataset),
                "train_samples": len(train_set),
                "val_samples": len(val_set),
                "action_dim_raw": dataset.action_dim,
                "effective_action_dim": dataset.action_dim * frameskip,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "train_summary": summary,
            },
            f,
            indent=2,
        )