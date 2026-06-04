import importlib.machinery
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def install_stubs():
    if "loralib" not in sys.modules:
        lora_stub = types.ModuleType("loralib")
        lora_stub.__spec__ = importlib.machinery.ModuleSpec("loralib", loader=None)
        sys.modules["loralib"] = lora_stub

    if "peft" not in sys.modules:
        peft_stub = types.ModuleType("peft")
        peft_stub.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None)

        class _DummyLoraConfig:
            def __init__(self, *args, **kwargs):
                pass

        def _dummy_get_peft_model(model, *args, **kwargs):
            return model

        peft_stub.LoraConfig = _DummyLoraConfig
        peft_stub.get_peft_model = _dummy_get_peft_model
        sys.modules["peft"] = peft_stub

    if "swanlab" not in sys.modules:
        swan_stub = types.ModuleType("swanlab")
        swan_stub.__spec__ = importlib.machinery.ModuleSpec("swanlab", loader=None)

        def _init(*args, **kwargs):
            return {"ok": True}

        def _log(*args, **kwargs):
            return None

        swan_stub.init = _init
        swan_stub.log = _log
        sys.modules["swanlab"] = swan_stub


class DummyEncoderContainer(nn.Module):
    def __init__(self, layers=2, dim=16, branch="protein"):
        super().__init__()
        self.encoder = nn.Module()
        if branch == "protein":
            self.encoder.layer = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])
        else:
            self.encoder.block = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])
        self.config = types.SimpleNamespace(num_hidden_layers=layers)


class DummyTower(nn.Module):
    def __init__(self, layers=2, dim=16, branch="protein"):
        super().__init__()
        self.backbone = DummyEncoderContainer(layers=layers, dim=dim, branch=branch)


class _MaskStub:
    def __init__(self):
        self.droprate = 0.0


class DummyKineticsPredictor(nn.Module):
    def __init__(self, device="cpu", esm_model=None, molt5_model=None, cfg=None, **kwargs):
        super().__init__()
        self.device = torch.device(device)
        self.cfg = cfg

        self.protein = DummyTower(layers=2, dim=16, branch="protein")
        self.ligand = DummyTower(layers=2, dim=16, branch="ligand")

        self.adapter = nn.Linear(1, 16)
        self.fusion = nn.Linear(16, 16)
        self.head_proj = nn.Linear(16, 16)
        self.head_kcat = nn.Linear(16, 1)
        self.head_km = nn.Linear(16, 1)
        self.gate_e_proj = nn.Linear(16, 8)
        self.gate_s_proj = nn.Linear(16, 8)
        self.interactions = nn.ModuleList([nn.ModuleList([nn.Identity()])])

        self.resmask = _MaskStub()

        self.s_kcat = nn.Parameter(torch.tensor(0.0))
        self.s_Km = nn.Parameter(torch.tensor(0.0))
        self.s_ratio = nn.Parameter(torch.tensor(0.0))

    def forward(self, seqs, smiles, use_mask=True, bias_tuple=(0.0, 0.0, 0.0), caps=(0.5, 0.5, 0.4), floor_m=-1e9, prot_struct=None, lig_struct=None):
        lengths = torch.tensor([float(len(s)) for s in seqs], device=self.device).unsqueeze(1)
        x = self.adapter(lengths / 10.0)
        x = torch.tanh(self.fusion(x))
        h = torch.relu(self.head_proj(x))

        pred_kcat = self.head_kcat(h).squeeze(-1)
        pred_km = self.head_km(h).squeeze(-1)
        pred_ratio = pred_kcat - pred_km + 3.0

        gate_e = torch.sigmoid(self.gate_e_proj(h))
        gate_s = torch.sigmoid(self.gate_s_proj(h))

        b_k, b_m, b_r = bias_tuple
        cap_k, cap_m, cap_r = caps
        s_k_eff = torch.clamp(self.s_kcat + b_k, min=-1e9, max=cap_k)
        s_m_eff = torch.clamp(self.s_Km + b_m, min=floor_m, max=cap_m)
        s_r_eff = torch.clamp(self.s_ratio + b_r, min=-1e9, max=cap_r)

        s_raw = (self.s_kcat, self.s_Km, self.s_ratio)
        s_eff = (s_k_eff, s_m_eff, s_r_eff)
        return pred_kcat, pred_km, pred_ratio, gate_e, gate_s, s_raw, s_eff


def write_tiny_dataset(csv_path):
    rows = [
        ("MKTLLILAV", "CCO", 1.20, 0.00012),
        ("AGGTTPLVA", "CCC", 0.85, 0.00020),
        ("VVVAAAGGG", "CCN", 2.10, 0.00008),
        ("LIPSTQNRK", "COC", 1.75, 0.00015),
        ("MNKQWERTY", "CCCl", 0.95, 0.00022),
        ("PPQQRRTTS", "CCBr", 1.40, 0.00018),
        ("GHIKLMNPQ", "CC(C)O", 2.35, 0.00009),
        ("RSTVWYAAA", "CC(C)N", 1.05, 0.00025),
        ("ACDEFGHIK", "CCOC", 1.90, 0.00011),
        ("LMNPQRSTV", "CCCN", 0.70, 0.00030),
        ("WYACDEFGH", "CNC", 2.60, 0.00007),
        ("IKLMNPQRS", "COCC", 1.30, 0.00017),
        ("TVWYACDEF", "CC(C)Cl", 0.88, 0.00021),
        ("GHIKLMNPA", "CC(C)Br", 1.55, 0.00016),
        ("QRSTVWYAC", "CCOCC", 2.05, 0.00010),
        ("DEFGHIKLM", "CCNCC", 1.12, 0.00024),
        ("NPQRSTVWY", "COCN", 1.68, 0.00014),
        ("AACCGGTTA", "CCS", 0.92, 0.00023),
        ("TTGGCCAAN", "CCF", 2.22, 0.00009),
        ("QWERTYUIO", "CCP", 1.48, 0.00019),
    ]
    header = "Sequence,Smiles,kcat(s^-1),Km(M)\n"
    lines = [header] + [f"{s},{m},{k},{km}\n" for s, m, k, km in rows]
    csv_path.write_text("".join(lines), encoding="utf-8")


def main():
    install_stubs()
    os.environ.pop("WORLD_SIZE", None)
    os.environ.pop("RANK", None)
    os.environ.pop("LOCAL_RANK", None)

    import main_train_predictor_multigpu as mt

    mt.KineticsPredictor = DummyKineticsPredictor
    mt.AutoConfig.from_pretrained = lambda *args, **kwargs: types.SimpleNamespace(num_hidden_layers=2)
    mt.save_training_checkpoint = lambda *args, **kwargs: None

    base_dataloader = mt.DataLoader

    def _safe_dataloader(*args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs["pin_memory"] = False
        return base_dataloader(*args, **kwargs)

    mt.DataLoader = _safe_dataloader

    tiny_csv = ROOT / "kcat-over-Km-data_0.4simi-10fold.csv"
    write_tiny_dataset(tiny_csv)

    sys.argv = [
        "main_train_predictor_multigpu.py",
        "-csv",
        str(tiny_csv),
        "--run_name",
        "tiny_multigpu_smoke",
        "-epochs",
        "1",
        "-batch_size",
        "4",
        "-device",
        "cpu",
    ]
    mt.main()


if __name__ == "__main__":
    main()
