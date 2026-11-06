import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model, mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mult, d_model),
        )

    def forward(self, x):
        return self.net(x)

class CrossAttentionBlock(nn.Module):
    """
    双向 Cross-Attention Block:
      1. protein <- cross-attend -> ligand
      2. ligand <- cross-attend -> protein
      3. 输出蛋白和配体的更新表示
      4. 保存对齐矩阵 & orthogonality penalty & pocket loss
    """
    def __init__(self, d_model=768, num_heads=8, dropout=0.1, ffn_hidden=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_hidden = ffn_hidden or (d_model * 4)
        self.dropout = dropout

        self.pa = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.la = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        self.ln_p1 = nn.LayerNorm(d_model)
        self.ln_l1 = nn.LayerNorm(d_model)
        self.ffp = FeedForward(d_model, dropout=dropout)
        self.ffl = FeedForward(d_model, dropout=dropout)
        self.ln_p2 = nn.LayerNorm(d_model)
        self.ln_l2 = nn.LayerNorm(d_model)

        self.ligand_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        self.last_alignment_p2l = None
        self.last_alignment_l2p = None
        self.last_ortho = None
        self.last_pocket_loss = None

    def forward(self, p, l, p_mask=None, l_mask=None):
        kp = (~l_mask) if l_mask is not None else None
        kl = (~p_mask) if p_mask is not None else None

        p_ln = self.ln_p1(p)
        l_ln = self.ln_l1(l)

        attn_p, w_p2l = self.pa(query=p_ln, key=l_ln, value=l_ln, key_padding_mask=kp, need_weights=False)
        attn_p = F.dropout(attn_p, p=self.dropout, training=self.training)
        p = p + attn_p
        p = self.ln_p2(p + self.ffp(p))

        attn_l, w_l2p = self.la(query=l_ln, key=p_ln, value=p_ln, key_padding_mask=kl, need_weights=False)
        attn_l = F.dropout(attn_l, p=self.dropout, training=self.training)
        l = l + attn_l
        l = self.ln_l2(l + self.ffl(l))

        ligand_gate = self.ligand_gate(l)  # (B, Ll, 1)
        l = l * ligand_gate

        self.last_alignment_p2l = None#w_p2l.detach()  # (B, heads, Lp, Ll)
        self.last_alignment_l2p = None#w_l2p.detach()  # (B, heads, Ll, Lp)

        try:
            B, Lp, C = p.shape
            _, Ll, _ = l.shape
            p_center = p - p.mean(dim=1, keepdim=True)
            l_center = l - l.mean(dim=1, keepdim=True)
            M = torch.matmul(p_center.transpose(1, 2), l_center) / (Lp * Ll + 1e-12)
            frob = torch.sqrt((M ** 2).sum(dim=(1, 2)) + 1e-12)
            np_norm = torch.sqrt((p_center ** 2).sum(dim=(1, 2)) + 1e-12)
            nl_norm = torch.sqrt((l_center ** 2).sum(dim=(1, 2)) + 1e-12)
            self.last_ortho = (frob / (np_norm * nl_norm + 1e-12)).mean().detach()
        except Exception:
            self.last_ortho = torch.tensor(0.0, device=p.device)

        if self.last_alignment_p2l is not None:
            p2l = self.last_alignment_p2l.mean(dim=1)  # 平均多头注意力: (B, Lp, Ll)
            self.last_pocket_loss = -torch.sum(p2l * torch.log(p2l + 1e-12), dim=-1).mean()
        else:
            self.last_pocket_loss = torch.tensor(0.0, device=p.device)

        return p, l

class ResidueMask(nn.Module):
    def __init__(self, d_model=768, temperature=2/3, stretch=(-0.1, 1.1), droprate=0.5):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        self.temperature = temperature
        self.gamma, self.zeta = stretch
        self.droprate = droprate

    def _hard_concrete(self, logits):
        u = torch.rand_like(logits)
        s = torch.sigmoid((logits + torch.log(u) - torch.log(1 - u)) / self.temperature)
        s = s * (self.zeta - self.gamma) + self.gamma
        s = s.clamp(0, 1)
        return s

    def forward(self, p_tok, train_mask=True):
        logits = self.score(p_tok)
        if self.training and train_mask:
            gate = self._hard_concrete(logits)
        else:
            gate = torch.sigmoid(logits).clamp(0, 1)
        erase = (gate * self.droprate)
        return gate.squeeze(-1), erase




class LigandMask(nn.Module):
    """
    Ligand-side gating module.
    输入:  l_tok (B, Ll, D)
    输出: gate_s (B, Ll), erase_s (B, Ll, 1)
    """
    def __init__(self, d_model=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # 输出每个 token 的一个 gate logit
        )

    def forward(self, l_tok, train_mask=True):
        """
        Args:
            l_tok: (B, Ll, D)
            train_mask: bool, 是否启用 gating mask
        Returns:
            gate_s: (B, Ll) ligand token gate ∈ [0,1]
            erase_s: (B, Ll, 1) mask for l_tok, same ∈ [0,1]
        """
        logits = self.proj(l_tok)  # (B, Ll, 1)
        gate_s = torch.sigmoid(logits).squeeze(-1)  # (B, Ll)

        erase_s = torch.sigmoid(logits)  # (B, Ll, 1)

        if train_mask and self.training:
            erase_s = self.dropout(erase_s)

        return gate_s, erase_s
