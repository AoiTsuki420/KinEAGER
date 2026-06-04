#!/usr/bin/env bash
set -euo pipefail

if [ -d /root/autodl-tmp/gptsrc_ddp ]; then
  export ROOT=/root/autodl-tmp/gptsrc_ddp
elif [ -d /root/autodl-tmp/gptsrc ]; then
  export ROOT=/root/autodl-tmp/gptsrc
else
  echo "[ERR] cannot find repo root"; exit 1
fi

export DATA_ROOT=${DATA_ROOT:-/autodl-fs/data/runs/splits/splits_revision}
export WORK_ROOT=${WORK_ROOT:-/root/autodl-tmp/splits_revision_with_struct}
export PDB_JSON=${PDB_JSON:-/root/autodl-fs/kcat_max_wt_singleSeqs_wpdbs_pdbrecords.json}
export PDB_OUT=${PDB_OUT:-/root/autodl-fs/struct_cache/pdbs}
export SDF_OUT=${SDF_OUT:-/root/autodl-fs/struct_cache/sdfs}
export OURS_ENV=${OURS_ENV:-}

run_py () {
  if [ -n "$OURS_ENV" ]; then
    conda run --no-capture-output -n "$OURS_ENV" python "$@"
  else
    python "$@"
  fi
}

[ -f "$PDB_JSON" ] || { echo "[ERR] PDB_JSON not found: $PDB_JSON"; exit 1; }
[ -d "$DATA_ROOT" ] || { echo "[ERR] DATA_ROOT not found: $DATA_ROOT"; exit 1; }
[ -f "$ROOT/tools/extract_pdb_from_catpred_json.py" ] || { echo "[ERR] missing tool: extract_pdb_from_catpred_json.py"; exit 1; }
[ -f "$ROOT/tools/extract_lig_sdf_from_smiles.py" ] || { echo "[ERR] missing tool: extract_lig_sdf_from_smiles.py"; exit 1; }
mkdir -p "$PDB_OUT" "$SDF_OUT"

PROBE_FILE="$DATA_ROOT/.write_probe.$$"
if touch "$PROBE_FILE" 2>/dev/null; then
  rm -f "$PROBE_FILE"
  CSV_ROOT="$DATA_ROOT"
  echo "[ok] DATA_ROOT is writable, modifying csvs in place: $DATA_ROOT"
else
  echo "[info] DATA_ROOT is read-only ($DATA_ROOT), mirroring to WORK_ROOT: $WORK_ROOT"
  mkdir -p "$WORK_ROOT"
  for it in iteration_{1..10}; do
    [ -d "$DATA_ROOT/$it" ] || continue
    mkdir -p "$WORK_ROOT/$it"
    for f in "$DATA_ROOT/$it"/kcat_test_split_*.csv \
             "$DATA_ROOT/$it"/kcat-seq_test_sequence_99cluster.csv \
             "$DATA_ROOT/$it"/kcat-seq_test_sequence_80cluster.csv \
             "$DATA_ROOT/$it"/kcat-seq_test_sequence_60cluster.csv \
             "$DATA_ROOT/$it"/kcat-seq_test_sequence_40cluster.csv; do
      [ -f "$f" ] || continue
      tgt="$WORK_ROOT/$it/$(basename "$f")"
      if [ ! -f "$tgt" ]; then cp "$f" "$tgt"; fi
    done
  done
  CSV_ROOT="$WORK_ROOT"
  echo "[ok] mirrored, will modify in: $CSV_ROOT"
  echo "[note] 推理脚本里需要把 DATA_ROOT 也指向这里:"
  echo "       export DATA_ROOT=$CSV_ROOT"
fi

mapfile -t TEST_CSVS < <(
  for it in iteration_{1..10}; do
    [ -d "$CSV_ROOT/$it" ] || continue
    for f in "$CSV_ROOT/$it"/kcat_test_split_*.csv \
             "$CSV_ROOT/$it"/kcat-seq_test_sequence_99cluster.csv \
             "$CSV_ROOT/$it"/kcat-seq_test_sequence_80cluster.csv \
             "$CSV_ROOT/$it"/kcat-seq_test_sequence_60cluster.csv \
             "$CSV_ROOT/$it"/kcat-seq_test_sequence_40cluster.csv; do
      [ -f "$f" ] && echo "$f"
    done
  done
)
echo "[scan] found ${#TEST_CSVS[@]} test csvs"
[ ${#TEST_CSVS[@]} -gt 0 ] || { echo "[ERR] no csvs found under $CSV_ROOT"; exit 1; }

echo "===== Step 1/2: extract PDB from JSON, fill prot_pdb_path ====="
run_py "$ROOT/tools/extract_pdb_from_catpred_json.py" \
  --json "$PDB_JSON" \
  --csvs "${TEST_CSVS[@]}" \
  --pdb_out_dir "$PDB_OUT" \
  --seq_col sequence \
  --match_by seq \
  --pdb_path_col prot_pdb_path \
  --skip_if_pdb_exists \
  --inplace

echo "===== Step 2/2: ETKDG 3D embed reactant_smiles, fill lig_sdf_path ====="
run_py "$ROOT/tools/extract_lig_sdf_from_smiles.py" \
  --csvs "${TEST_CSVS[@]}" \
  --sdf_out_dir "$SDF_OUT" \
  --smiles_col reactant_smiles \
  --lig_path_col lig_sdf_path \
  --inplace

echo "===== Verify: sample iteration_1/test ====="
SAMPLE="$CSV_ROOT/iteration_1/$(basename $(ls $CSV_ROOT/iteration_1/kcat_test_split_*.csv | head -n1))"
run_py - <<PY
import pandas as pd
df = pd.read_csv("$SAMPLE")
print(f"[verify] $SAMPLE: rows={len(df)}")
print(f"  prot_pdb_path filled: {df['prot_pdb_path'].notna().sum()}/{len(df)}")
print(f"  lig_sdf_path  filled: {df['lig_sdf_path'].notna().sum()}/{len(df)}")
both = (df['prot_pdb_path'].notna() & df['lig_sdf_path'].notna()).sum()
print(f"  both filled (has_struct=True): {both}/{len(df)} = {100.0*both/len(df):.1f}%")
PY

echo "[done] PDBs in $PDB_OUT, SDFs in $SDF_OUT"
echo "       50 csvs were updated in place; expert/MoE pipeline now ready."
echo ""
echo ">>> 下一步推理:"
echo "       export DATA_ROOT=$CSV_ROOT"
echo "       MODE=moe_softgate bash $ROOT/run_ours_10iter_inference.sh"
