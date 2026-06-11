"""Mock smoke test for KinEAGER + OODRouter.

Validates the fusion + routing logic without requiring a real expert checkpoint.
Loads the real precomputed train index (runs/moe_index_smoke/train_emb.npy).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.moe_kcat import KinEAGER, OODRouter, build_router_from_npy  # noqa: E402



class MockMain(nn.Module):
    """Returns (pred_kcat, pred_km, pred_ratio)-like tuple; only [0] is used."""

    def __init__(self, base: float = 1.0, noise: float = 0.01):
        super().__init__()
        self.base = base
        self.noise = noise
        self.dummy = nn.Linear(1, 1)  # ensures .parameters() non-empty

    def forward(self, seqs, smiles, source_ids=None, use_mask=False):
        B = len(seqs)
        mu = torch.full((B,), self.base, device=self.dummy.weight.device)
        mu = mu + self.noise * torch.randn(B, device=mu.device)
        return (mu,)


class MockExpert(nn.Module):
    """Returns (mu, log_var) from a batch dict."""

    def __init__(self, base: float = 2.0, noise: float = 0.01):
        super().__init__()
        self.base = base
        self.noise = noise
        self.dummy = nn.Linear(1, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, batch):
        B = len(batch["seqs"])
        dev = self.dummy.weight.device
        mu = torch.full((B,), self.base, device=dev)
        mu = mu + self.noise * self.dropout(torch.ones(B, device=dev))
        log_var = torch.full((B,), -2.0, device=dev)
        return mu, log_var


class MockEsmEncoder(nn.Module):
    """Fake ProteinEncoder: encode_tokens(seqs) -> (hidden [B,L,D], mask [B,L])."""

    def __init__(self, hidden_dim: int, seed_base: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seed_base = seed_base
        self.dummy = nn.Linear(1, 1)  # so .parameters() not empty

    def encode_tokens(self, seqs: list[str]):
        dev = self.dummy.weight.device
        B = len(seqs)
        L = max(1, max(len(s) for s in seqs))
        hidden = []
        for i, s in enumerate(seqs):
            g = torch.Generator(device="cpu").manual_seed(abs(hash(s)) % (2 ** 31))
            v = torch.randn(L, self.hidden_dim, generator=g)
            hidden.append(v)
        hidden = torch.stack(hidden, dim=0).to(dev)
        mask = torch.zeros(B, L, dtype=torch.bool, device=dev)
        for i, s in enumerate(seqs):
            mask[i, : len(s)] = True
        return hidden, mask



def make_batch(seqs, smiles, has_struct_list, device, y_kcat=None):
    B = len(seqs)
    batch = {
        "seqs": seqs,
        "smiles": smiles,
        "prot_dist": torch.zeros(B, 1, 1, device=device),
        "prot_struct_mask": torch.zeros(B, 1, dtype=torch.bool, device=device),
        "lig_atom_feat": torch.zeros(B, 1, 44, device=device),
        "lig_atom_dist": torch.zeros(B, 1, 1, device=device),
        "lig_atom_mask": torch.zeros(B, 1, dtype=torch.bool, device=device),
        "geom_feats": torch.zeros(B, 16, device=device),
        "has_struct": torch.tensor(has_struct_list, dtype=torch.bool, device=device),
    }
    if y_kcat is not None:
        batch["y_kcat"] = torch.tensor(y_kcat, dtype=torch.float32, device=device)
    return batch


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        raise AssertionError(msg)



def test_router_monotonic(device):
    print("\n=== test_router_monotonic ===")
    D = 8
    idx = torch.eye(D)[:3]  # 3 orthogonal unit vectors
    router = OODRouter(idx, d0=0.5, tau=0.1, k=1).to(device)

    q = torch.stack([idx[0], torch.zeros(D)])
    q[1, D // 2] = 1.0  # orthogonal axis not in index
    q = q.to(device)
    w, d = router.route(q)
    print(f"  dist={d.tolist()}  w_expert={w.tolist()}")
    check(d[0].item() < 1e-5, "identical query -> distance ~0")
    check(abs(d[1].item() - 1.0) < 1e-5, "orthogonal query -> distance 1")
    check(w[0].item() < 0.05, "low OOD -> small w_expert")
    check(w[1].item() > 0.95, "high OOD -> large w_expert")
    check(w[1].item() > w[0].item(), "w_expert monotonic in OOD distance")


def test_fusion_math(device):
    print("\n=== test_fusion_math ===")
    D = 16
    idx = torch.eye(D)[:2]
    router = OODRouter(idx, d0=0.5, tau=0.05, k=1).to(device)

    main = MockMain(base=1.0, noise=0.0).to(device)
    expert = MockExpert(base=3.0, noise=0.0).to(device)
    q_enc = MockEsmEncoder(hidden_dim=D).to(device)

    moe = KinEAGER(main, expert, router, q_enc,
                  main_mc_samples=3, expert_mc_samples=3,
                  hard_gate_on_no_struct=True,
                  use_precision_weighting=False,
                  source_id=0).to(device)

    batch = make_batch(
        seqs=["SEQA", "SEQB"],
        smiles=["CCO", "CCN"],
        has_struct_list=[True, True],
        device=device,
    )
    out = moe.predict(batch)

    check(set(out.keys()) >= {"mu_main", "mu_expert", "ood_score", "w_main", "w_expert", "mu_ensemble", "s2_ensemble"},
          "predict returns all expected keys")

    mu_m = out["mu_main"]; mu_e = out["mu_expert"]
    w_m = out["w_main"]; w_e = out["w_expert"]; mu_ens = out["mu_ensemble"]
    check(torch.allclose(w_m + w_e, torch.ones_like(w_m), atol=1e-6),
          "w_main + w_expert == 1")
    expected = w_m * mu_m + w_e * mu_e
    check(torch.allclose(mu_ens, expected, atol=1e-6),
          "mu_ensemble == w_main*mu_main + w_expert*mu_expert")
    check(torch.allclose(mu_m, torch.full_like(mu_m, 1.0), atol=1e-4),
          "mu_main close to mock base 1.0")
    check(torch.allclose(mu_e, torch.full_like(mu_e, 3.0), atol=0.2),
          "mu_expert close to mock base 3.0 (dropout adds small perturbation)")
    print(f"  mu_main={mu_m.tolist()}  mu_expert={mu_e.tolist()}")
    print(f"  ood={out['ood_score'].tolist()}  w_expert={w_e.tolist()}")
    print(f"  mu_ensemble={mu_ens.tolist()}")


def test_hard_gate_no_struct(device):
    print("\n=== test_hard_gate_no_struct ===")
    D = 16
    idx = torch.zeros(3, D)  # makes every query maximally OOD -> w_expert→1
    idx[:, 0] = 1.0          # all index points are the same, doesn't matter
    router = OODRouter(idx, d0=0.0, tau=0.01, k=1).to(device)

    main = MockMain(base=1.0, noise=0.0).to(device)
    expert = MockExpert(base=5.0, noise=0.0).to(device)
    q_enc = MockEsmEncoder(hidden_dim=D).to(device)

    moe = KinEAGER(main, expert, router, q_enc,
                  main_mc_samples=1, expert_mc_samples=1,
                  hard_gate_on_no_struct=True,
                  use_precision_weighting=False).to(device)

    batch = make_batch(
        seqs=["SEQX", "SEQY"],
        smiles=["C", "N"],
        has_struct_list=[True, False],
        device=device,
    )
    out = moe.predict(batch)
    w_e = out["w_expert"]
    print(f"  has_struct=[T,F]  w_expert={w_e.tolist()}")
    check(w_e[1].item() == 0.0, "has_struct=False -> w_expert forced to 0")
    check(w_e[0].item() > 0.3, "has_struct=True + high OOD -> w_expert > 0.3")
    check(torch.allclose(out["mu_ensemble"][1], out["mu_main"][1], atol=1e-6),
          "no-struct sample -> mu_ensemble == mu_main")


def test_precision_weighting(device):
    print("\n=== test_precision_weighting ===")
    D = 16
    idx = torch.eye(D)[:2]
    router = OODRouter(idx, d0=0.5, tau=0.05, k=1).to(device)

    main = MockMain(base=1.0, noise=0.0).to(device)

    class NoisyExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Linear(1, 1)

        def forward(self, batch):
            B = len(batch["seqs"])
            dev = self.dummy.weight.device
            return (torch.full((B,), 5.0, device=dev),
                    torch.full((B,), 4.0, device=dev))  # huge log_var -> huge s2

    expert = NoisyExpert().to(device)
    q_enc = MockEsmEncoder(hidden_dim=D).to(device)

    moe_off = KinEAGER(main, expert, router, q_enc,
                      main_mc_samples=2, expert_mc_samples=2,
                      hard_gate_on_no_struct=True,
                      use_precision_weighting=False).to(device)
    moe_on = KinEAGER(main, expert, router, q_enc,
                     main_mc_samples=2, expert_mc_samples=2,
                     hard_gate_on_no_struct=True,
                     use_precision_weighting=True).to(device)

    batch = make_batch(
        seqs=["HIGHLY_OOD_SEQ_ABC"] * 4,
        smiles=["CC"] * 4,
        has_struct_list=[True] * 4,
        device=device,
    )
    torch.manual_seed(0)
    out_off = moe_off.predict(batch)
    torch.manual_seed(0)
    out_on = moe_on.predict(batch)
    print(f"  w_expert (no precision) = {out_off['w_expert'].tolist()}")
    print(f"  w_expert (w/ precision) = {out_on['w_expert'].tolist()}")
    check((out_on["w_expert"] <= out_off["w_expert"] + 1e-6).all().item(),
          "high expert uncertainty + precision weighting -> w_expert not higher")


def test_router_from_real_npy(device):
    print("\n=== test_router_from_real_npy (loads real precompute output) ===")
    npy = REPO / "runs" / "moe_index_smoke" / "train_emb.npy"
    if not npy.exists():
        print(f"  [SKIP] index not found at {npy}")
        return
    router = build_router_from_npy(str(npy), d0=0.15, tau=0.05, k=1, device=device)
    N, D = router.train_emb.shape
    print(f"  loaded index shape=[{N}, {D}]")
    check(N > 0 and D > 0, "index non-empty")
    norms = router.train_emb.norm(dim=-1)
    check(torch.allclose(norms, torch.ones_like(norms), atol=1e-4),
          "index rows are L2-normalized")
    q = router.train_emb[:1]
    w, d = router.route(q)
    check(d[0].item() < 1e-4, "self-query gives ~0 distance")
    check(w[0].item() < 0.5, "self-query gives small w_expert at d0=0.15")



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    torch.manual_seed(42)

    test_router_monotonic(device)
    test_fusion_math(device)
    test_hard_gate_no_struct(device)
    test_precision_weighting(device)
    test_router_from_real_npy(device)

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
