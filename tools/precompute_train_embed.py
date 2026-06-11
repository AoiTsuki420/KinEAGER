"""为 KinEAGER 的 OOD router 预计算 expert 训练集序列的 ESM 平均池化 embedding。

输出一个 .npy（float32，已 L2 归一化）+ 一个 .csv（uniq_id → sequence 映射）。
用最近邻余弦距离做 OOD 路由时，直接加载 .npy 即可。

用法:
  python tools/precompute_train_embed.py \
      --train_csv /root/autodl-fs/itera/train_merged_with_kcat_geom_clean.csv \
      --esm_path  /root/autodl-fs/models/esm2_t33_650M_UR50D \
      --out_dir   runs/moe_index \
      --batch_size 8 --max_len 1024

备注:
  - 同一条序列（不同底物/拷贝）会被去重，只算一次。
  - 输出 .npy 已 L2 归一化，配合矩阵乘 q @ train_emb.T 即为余弦相似度。
  - 默认只读 'sequence' 列；如 CSV 用 'Sequence'，传 --rename_cols default。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


DEFAULT_RENAME = {
    "Sequence": "sequence",
    "Substrate SMILES": "smiles",
    "pkcat_value": "kcat_log10",
}


def _load_esm(esm_path: str, device: str):
    try:
        from transformers import EsmTokenizer, EsmModel
        tok = EsmTokenizer.from_pretrained(esm_path, local_files_only=False)
        mdl = EsmModel.from_pretrained(esm_path, local_files_only=False)
    except Exception:
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(esm_path, local_files_only=False)
        mdl = AutoModel.from_pretrained(esm_path, local_files_only=False)
    mdl = mdl.to(device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    return tok, mdl


def _pool(hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    m = attn_mask.float().unsqueeze(-1)
    return (hidden * m).sum(1) / m.sum(1).clamp_min(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--esm_path", required=True,
                    help="与 expert 训练时一致的 ESM2 路径（建议 650M）")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seq_col", default=None,
                    help="显式指定序列列名；不填则自动探测 sequence/Sequence/seq")
    ap.add_argument("--rename_cols", default="none",
                    help='default | none | 自定义 JSON 字符串')
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "train_emb.npy"
    meta_path = out_dir / "train_meta.csv"
    info_path = out_dir / "train_emb_info.json"
    if emb_path.exists() and meta_path.exists() and not args.overwrite:
        print(f"[skip] outputs already exist under {out_dir} (use --overwrite to regenerate)")
        return

    df = pd.read_csv(args.train_csv)
    if args.rename_cols == "default":
        df = df.rename(columns=DEFAULT_RENAME)
    elif args.rename_cols == "none":
        pass
    else:
        df = df.rename(columns=json.loads(args.rename_cols))

    seq_col = args.seq_col
    if seq_col is None:
        for cand in ("sequence", "Sequence", "seq"):
            if cand in df.columns:
                seq_col = cand
                break
    if seq_col is None:
        raise ValueError(f"no sequence column found. columns={list(df.columns)}")

    raw = df[seq_col].astype(str).str.strip().str.upper().tolist()
    uniq, seen = [], {}
    for s in raw:
        if not s or s == "NAN":
            continue
        if s not in seen:
            seen[s] = len(uniq)
            uniq.append(s)
    print(f"[precompute] rows={len(raw)} -> unique sequences={len(uniq)}")

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[precompute] loading ESM from {args.esm_path} on {device}")
    tokenizer, backbone = _load_esm(args.esm_path, device)
    hidden_dim = backbone.config.hidden_size
    print(f"[precompute] hidden_size={hidden_dim}")

    embs = np.empty((len(uniq), hidden_dim), dtype=np.float32)
    for i in tqdm(range(0, len(uniq), args.batch_size), desc="ESM encode"):
        chunk = [s[: args.max_len] for s in uniq[i : i + args.batch_size]]
        inputs = tokenizer(chunk, return_tensors="pt", padding=True,
                           truncation=True, max_length=args.max_len)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = backbone(**inputs).last_hidden_state
        mask = inputs["attention_mask"].bool()
        emb = _pool(out, mask)
        emb = F.normalize(emb, dim=-1)
        embs[i : i + len(chunk)] = emb.detach().cpu().float().numpy()

    np.save(emb_path, embs)
    pd.DataFrame({"uniq_id": list(range(len(uniq))), "sequence": uniq}).to_csv(meta_path, index=False)
    with open(info_path, "w") as f:
        json.dump({
            "esm_path": args.esm_path,
            "train_csv": args.train_csv,
            "n_unique": len(uniq),
            "n_rows": len(raw),
            "hidden_size": hidden_dim,
            "max_len": args.max_len,
            "seq_col": seq_col,
            "normalized": "l2",
        }, f, indent=2)

    print(f"[saved] {emb_path}  shape={embs.shape}  dtype={embs.dtype}")
    print(f"[saved] {meta_path}  rows={len(uniq)}")
    print(f"[saved] {info_path}")


if __name__ == "__main__":
    main()
