"""从 CatPred-DB 的 *_pdbrecords.json 抽出 .pdb 文件，并回填到测试 CSV。

JSON 格式（CatPred-DB 原生）:
  {
    "AF-<UniProt>-F1-model_v4.pdb": {
      "seq":   "MAAF...KS",
      "coords": [[[Nxyz],[CAxyz],[Cxyz],[Oxyz]], ...]  # per-residue
    },
    ...
  }

用法:
  python tools/extract_pdb_from_catpred_json.py \
      --json  /root/autodl-fs/.../kcat_max_wt_singleSeqs_wpdbs_pdbrecords.json \
      --csvs  /root/autodl-fs/itera/test19_ready.csv \
              /root/autodl-fs/itera/test40_ready.csv \
              /root/autodl-fs/itera/test60_ready.csv \
      --pdb_out_dir /root/autodl-fs/itera/catpred_pdbs \
      --seq_col sequence \
      --match_by seq

默认按序列精确匹配（seq 归一化：去空白 + 大写）。
若 CSV 已有 UniProt ID 列，可用 --match_by uniprot --uniprot_col uniprot_id。

输出:
  1) 每个用到的键写 1 个 .pdb 到 --pdb_out_dir/<key>
  2) CSV 原地更新，追加 prot_pdb_path 列（已存在且非空则跳过）
     保存为 <原名>.with_pdb.csv（加 --inplace 则覆盖原文件）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


AA1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    "X": "UNK",
}

BACKBONE_ATOMS = ["N", "CA", "C", "O"]
BACKBONE_ELEM = ["N", "C", "C", "O"]


def _seq_norm(s: str) -> str:
    return "".join(str(s).split()).upper()


def _write_pdb(pdb_path: Path, seq: str, coords, pdb_key: str):
    """写标准 ATOM 记录；每个残基 N/CA/C/O 四个原子。

    coords: list of [[N],[CA],[C],[O]]，len == len(seq)
    """
    n_res = min(len(seq), len(coords))
    atom_serial = 1
    lines = []
    for i in range(n_res):
        aa1 = seq[i]
        resname = AA1_TO_3.get(aa1, "UNK")
        resseq = i + 1
        for a_idx, atom_name in enumerate(BACKBONE_ATOMS):
            try:
                x, y, z = coords[i][a_idx]
            except Exception:
                continue
            elem = BACKBONE_ELEM[a_idx]
            line = (
                f"ATOM  "                                  # 1-6
                f"{atom_serial:>5d} "                      # 7-11 + space
                f" {atom_name:<3s}"                        # 13-16 (atom name, left-aligned in 4 chars)
                f" "                                       # 17 altLoc
                f"{resname:>3s}"                           # 18-20
                f" A"                                      # 21 space + chain A
                f"{resseq:>4d}    "                        # 23-26 + icode + 3 spaces
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"             # 31-54
                f"{1.00:>6.2f}{0.00:>6.2f}"                # 55-66 occupancy + b-factor
                f"          "                              # 67-76
                f"{elem:>2s}  "                            # 77-78
            )
            lines.append(line)
            atom_serial += 1
    lines.append("TER")
    lines.append("END")

    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    with pdb_path.open("w", encoding="utf-8") as f:
        f.write("REMARK   1 Generated from CatPred-DB pdbrecords JSON\n")
        f.write(f"REMARK   1 SOURCE_KEY {pdb_key}\n")
        for ln in lines:
            f.write(ln + "\n")


def _uniprot_from_key(key: str) -> str | None:
    stem = key.replace(".pdb", "")
    parts = stem.split("-")
    if len(parts) >= 2 and parts[0] == "AF":
        return parts[1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="CatPred-DB *_pdbrecords.json")
    ap.add_argument("--csvs", nargs="+", required=True, help="一个或多个测试 CSV")
    ap.add_argument("--pdb_out_dir", required=True, help="抽出的 .pdb 放这里")
    ap.add_argument("--seq_col", default="sequence")
    ap.add_argument("--match_by", choices=["seq", "uniprot"], default="seq")
    ap.add_argument("--uniprot_col", default="uniprot_id")
    ap.add_argument("--pdb_path_col", default="prot_pdb_path",
                    help="CSV 中写入的列名")
    ap.add_argument("--overwrite_existing_paths", action="store_true",
                    help="若 CSV 里 prot_pdb_path 已有非空值，也强制用 JSON 匹配到的覆盖")
    ap.add_argument("--inplace", action="store_true",
                    help="直接覆盖原 CSV；不加则另存为 <名>.with_pdb.csv")
    ap.add_argument("--skip_if_pdb_exists", action="store_true",
                    help="若目标 .pdb 已在 pdb_out_dir 存在则不重写（省 IO）")
    args = ap.parse_args()

    pdb_out_dir = Path(args.pdb_out_dir).resolve()
    pdb_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] JSON: {args.json}")
    with open(args.json, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"[load] records: {len(records)}")

    seq_to_key: dict[str, str] = {}
    uniprot_to_key: dict[str, str] = {}
    dup_seq = 0
    for key, rec in records.items():
        seq = rec.get("seq", "")
        if seq:
            sn = _seq_norm(seq)
            if sn in seq_to_key and seq_to_key[sn] != key:
                dup_seq += 1
            else:
                seq_to_key[sn] = key
        up = _uniprot_from_key(key)
        if up:
            uniprot_to_key[up.upper()] = key
    print(f"[index] unique seqs: {len(seq_to_key)} (dup_ignored={dup_seq}), "
          f"uniprot: {len(uniprot_to_key)}")

    written_keys: set[str] = set()
    for csv_path in args.csvs:
        csv_path = Path(csv_path)
        print(f"\n[csv] {csv_path}")
        df = pd.read_csv(csv_path)
        n = len(df)

        if args.match_by == "seq":
            if args.seq_col not in df.columns:
                raise ValueError(f"{csv_path}: missing seq column {args.seq_col}")
            keys = df[args.seq_col].astype(str).map(_seq_norm).map(seq_to_key)
        else:
            if args.uniprot_col not in df.columns:
                raise ValueError(f"{csv_path}: missing uniprot column {args.uniprot_col}")
            keys = df[args.uniprot_col].astype(str).str.upper().map(uniprot_to_key)

        hits = keys.notna().sum()
        print(f"[match] {hits}/{n} rows matched by {args.match_by}")

        uniq_keys = set(keys.dropna().unique().tolist())
        new_written = 0
        for k in uniq_keys:
            out_path = pdb_out_dir / k  # 文件名里已含 .pdb
            if args.skip_if_pdb_exists and out_path.exists():
                continue
            if k in written_keys and out_path.exists():
                continue
            rec = records[k]
            _write_pdb(out_path, rec["seq"], rec["coords"], k)
            written_keys.add(k)
            new_written += 1
        print(f"[write] new .pdb files: {new_written}")

        abs_paths = keys.map(
            lambda k: str((pdb_out_dir / k).resolve()) if isinstance(k, str) else None
        )
        if args.pdb_path_col in df.columns and not args.overwrite_existing_paths:
            existing = df[args.pdb_path_col].astype("object")
            mask_empty = existing.isna() | (existing.astype(str).str.len() == 0) | \
                         (existing.astype(str).str.lower().isin(["nan", "none"]))
            df.loc[mask_empty, args.pdb_path_col] = abs_paths[mask_empty]
            filled = int(mask_empty.sum() - abs_paths[mask_empty].isna().sum())
            print(f"[fill] filled {filled} empty rows into existing column "
                  f"{args.pdb_path_col}")
        else:
            df[args.pdb_path_col] = abs_paths
            print(f"[fill] wrote column {args.pdb_path_col} ({abs_paths.notna().sum()} non-null)")

        out_csv = csv_path if args.inplace else csv_path.with_suffix(".with_pdb.csv")
        df.to_csv(out_csv, index=False)
        print(f"[save] {out_csv}")

    print(f"\n[done] total new .pdb files written: {len(written_keys)}")


if __name__ == "__main__":
    main()
