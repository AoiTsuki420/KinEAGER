#!/usr/bin/env bash
set -euo pipefail

if [ -d /root/autodl-tmp/gptsrc_ddp ]; then
  export ROOT=/root/autodl-tmp/gptsrc_ddp
elif [ -d /root/autodl-tmp/gptsrc ]; then
  export ROOT=/root/autodl-tmp/gptsrc
else
  echo "[ERR] cannot find repo root"; exit 1
fi

export DATA_ROOT_RO=${DATA_ROOT_RO:-/autodl-fs/data/runs/splits/splits_revision}
export DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/splits_revision_with_struct}
export PDB_OUT=${PDB_OUT:-/root/autodl-fs/struct_cache/pdbs}
export SDF_OUT=${SDF_OUT:-/root/autodl-fs/struct_cache/sdfs}
export PDB_JSON=${PDB_JSON:-/root/autodl-fs/kcat_max_wt_singleSeqs_wpdbs_pdbrecords.json}

export EXPERT_OUT_ROOT=${EXPERT_OUT_ROOT:-/root/autodl-tmp/expert_per_iter_leakfree}
export RESULTS_ROOT=${RESULTS_ROOT:-/root/autodl-tmp/runs/core5_splits_revision_kcat_leakfree}

export ESM_PATH=${ESM_PATH:-/root/autodl-fs/models/esm2_t33_650M_UR50D}
export MOLT5_PATH=${MOLT5_PATH:-/root/autodl-fs/models/molt5-base-smiles2caption}

export OURS_WEIGHTS=${OURS_WEIGHTS:-/root/autodl-fs/runs/id_full_seed42/seed42/predictor_best_ema.pt}
export OURS_SCALER=${OURS_SCALER:-/root/autodl-fs/runs/id_full_seed42/seed42/scaler_fold.json}

export EXPERT_CKPT_PRESET=${EXPERT_CKPT_PRESET:-$ROOT/runs/kcat_expert_v7/best.pt}
export TRAIN_EMB_NPY_PRESET=${TRAIN_EMB_NPY_PRESET:-$ROOT/runs/moe_index/train_emb.npy}

export INIT_CKPT_V6=${INIT_CKPT_V6:-$ROOT/runs/kcat_expert_v6/best.pt}

export EXPERT_EPOCHS=${EXPERT_EPOCHS:-10}
export EXPERT_BS=${EXPERT_BS:-24}
export EXPERT_GRAD_ACCUM=${EXPERT_GRAD_ACCUM:-4}
export EXPERT_LR=${EXPERT_LR:-8e-5}
export EXPERT_LORA_LR=${EXPERT_LORA_LR:-8e-5}
export EXPERT_LORA_RANK=${EXPERT_LORA_RANK:-32}
export EXPERT_LORA_ALPHA=${EXPERT_LORA_ALPHA:-64}
export EXPERT_WD=${EXPERT_WD:-5e-4}
export EXPERT_RANK_W=${EXPERT_RANK_W:-0.3}
export EXPERT_VAR_W=${EXPERT_VAR_W:-0.2}
export EXPERT_NLL_W=${EXPERT_NLL_W:-0.05}
export EXPERT_CCC_W=${EXPERT_CCC_W:-0.5}
export EXPERT_GRAD_CLIP=${EXPERT_GRAD_CLIP:-3.0}
export EXPERT_SEQ_CROP=${EXPERT_SEQ_CROP:-0.9}
export EXPERT_SEQ_MASK=${EXPERT_SEQ_MASK:-0.05}
export EXPERT_PATIENCE=${EXPERT_PATIENCE:-8}
export VAL_FRAC=${VAL_FRAC:-0.1}
export SEED=${SEED:-42}

RENAME_JSON='{"reactant_smiles":"smiles","log10kcat_max":"kcat_log10"}'

ITERS_TO_RETRAIN=${ITERS_TO_RETRAIN:-"5 9"}
ITERS_TO_INFERENCE=${ITERS_TO_INFERENCE:-"1 5 9"}
read -r -a RETRAIN_ITERS <<< "$ITERS_TO_RETRAIN"
read -r -a INFER_ITERS   <<< "$ITERS_TO_INFERENCE"

mkdir -p "$EXPERT_OUT_ROOT" "$RESULTS_ROOT"

in_retrain () {
  local n="$1"
  for x in "${RETRAIN_ITERS[@]}"; do [ "$x" = "$n" ] && return 0; done
  return 1
}

[ -f "$PDB_JSON" ]      || { echo "[ERR] PDB_JSON missing: $PDB_JSON"; exit 1; }
[ -d "$DATA_ROOT_RO" ]  || { echo "[ERR] DATA_ROOT_RO missing: $DATA_ROOT_RO"; exit 1; }
[ -f "$OURS_WEIGHTS" ]  || { echo "[ERR] OURS_WEIGHTS missing: $OURS_WEIGHTS"; exit 1; }
[ -f "$ROOT/main_train_kcat_expert.py" ] || { echo "[ERR] missing main_train_kcat_expert.py"; exit 1; }
[ -f "$ROOT/run_ours_10iter_inference.sh" ] || { echo "[ERR] missing run_ours_10iter_inference.sh"; exit 1; }

NEED_PRESET=0
for it in "${INFER_ITERS[@]}"; do
  if ! in_retrain "$it"; then
    [ -f "$EXPERT_CKPT_PRESET" ]  || { echo "[ERR] iter_${it} 不 retrain, 但 EXPERT_CKPT_PRESET 不存在: $EXPERT_CKPT_PRESET"; exit 1; }
    [ -f "$TRAIN_EMB_NPY_PRESET" ] || { echo "[ERR] iter_${it} 不 retrain, 但 TRAIN_EMB_NPY_PRESET 不存在: $TRAIN_EMB_NPY_PRESET"; exit 1; }
    NEED_PRESET=1
  fi
done

if [ $NEED_PRESET -eq 1 ]; then
  if ! TRAIN_EMB_NPY_PRESET="$TRAIN_EMB_NPY_PRESET" python - <<'PY'
import os, sys, numpy as np
p = os.environ["TRAIN_EMB_NPY_PRESET"]
e = np.load(p, mmap_mode="r")
print(f"[preflight] PRESET train_emb {p}  shape={tuple(e.shape)}  dtype={e.dtype}")
if e.ndim != 2 or e.shape[1] != 1280 or e.shape[0] < 100:
    print(f"[preflight] expect (N>=100, 1280) for ESM-650M, got {tuple(e.shape)}")
    sys.exit(2)
PY
  then
    echo ""
    echo "[ERR] PRESET train_emb 维度异常 — 不能用于 ESM-650M 推理。"
    echo "      重新生成 (示例, 用 v7 对应的 SKiD merged train csv):"
    echo "        python tools/precompute_train_embed.py \\"
    echo "          --train_csv /root/autodl-fs/itera/train_merged_with_kcat_geom_clean.csv \\"
    echo "          --esm_path  $ESM_PATH \\"
    echo "          --rename_cols default \\"
    echo "          --out_dir   $(dirname "$TRAIN_EMB_NPY_PRESET") \\"
    echo "          --batch_size 8 --max_len 1024 --overwrite"
    exit 1
  fi
fi

for it in "${INFER_ITERS[@]}"; do
  IT_DIR_RO="$DATA_ROOT_RO/iteration_${it}"
  [ -d "$IT_DIR_RO" ] || { echo "[ERR] DATA_ROOT_RO missing iteration dir: $IT_DIR_RO"; exit 1; }
  TEST_CSV_RO=$(ls "$IT_DIR_RO"/kcat_test_split_*.csv 2>/dev/null | head -n1)
  [ -f "$TEST_CSV_RO" ] || { echo "[ERR] iter_${it}: no kcat_test_split_*.csv under $IT_DIR_RO"; exit 1; }
  for sp in 99 80 60 40; do
    [ -f "$IT_DIR_RO/kcat-seq_test_sequence_${sp}cluster.csv" ] || \
      echo "[WARN] iter_${it}: missing kcat-seq_test_sequence_${sp}cluster.csv (会 [MISS] skip)"
  done
done

echo "[plan] retrain iters:   ${RETRAIN_ITERS[*]:-<none>}"
echo "[plan] inference iters: ${INFER_ITERS[*]}"
echo "[plan] preset expert:   $EXPERT_CKPT_PRESET"
echo "[plan] preset emb:      $TRAIN_EMB_NPY_PRESET"
echo "[plan] data root (RO):  $DATA_ROOT_RO   <-- inference 读 test csv 的位置"

if [ ${#RETRAIN_ITERS[@]} -gt 0 ]; then
  echo ""
  echo "########## Step 0: prep struct for train csvs (${RETRAIN_ITERS[*]}) ##########"
  TRAIN_CSVS=()
  for it in "${RETRAIN_ITERS[@]}"; do
    src_dir="$DATA_ROOT_RO/iteration_${it}"
    dst_dir="$DATA_ROOT/iteration_${it}"
    mkdir -p "$dst_dir"
    for f in "$src_dir"/kcat_train_split_*.csv; do
      [ -f "$f" ] || continue
      tgt="$dst_dir/$(basename "$f")"
      if [ ! -f "$tgt" ]; then
        cp "$f" "$tgt"
        echo "[copy] $f -> $tgt"
      fi
      TRAIN_CSVS+=("$tgt")
    done
  done

  if [ ${#TRAIN_CSVS[@]} -gt 0 ]; then
    python "$ROOT/tools/extract_pdb_from_catpred_json.py" \
      --json "$PDB_JSON" \
      --csvs "${TRAIN_CSVS[@]}" \
      --pdb_out_dir "$PDB_OUT" \
      --seq_col sequence --match_by seq \
      --pdb_path_col prot_pdb_path \
      --skip_if_pdb_exists --inplace

    python "$ROOT/tools/extract_lig_sdf_from_smiles.py" \
      --csvs "${TRAIN_CSVS[@]}" \
      --sdf_out_dir "$SDF_OUT" \
      --smiles_col reactant_smiles \
      --lig_path_col lig_sdf_path \
      --inplace
  fi
else
  echo "[Step 0 skipped — no iter to retrain]"
fi

for it in "${RETRAIN_ITERS[@]}"; do
  echo ""
  echo "##########################################################"
  echo "#  Step 1 / iter_${it}: retrain expert + emb"
  echo "##########################################################"

  IT_DIR="$DATA_ROOT/iteration_${it}"
  TRAIN_CSV=$(ls "$IT_DIR"/kcat_train_split_*.csv 2>/dev/null | head -n1)
  if [ -z "$TRAIN_CSV" ] || [ ! -f "$TRAIN_CSV" ]; then
    echo "[ERR] iter $it: no train csv under $IT_DIR — skip"
    continue
  fi

  EXP_OUT="$EXPERT_OUT_ROOT/iter_${it}"
  mkdir -p "$EXP_OUT"

  if [ ! -f "$EXP_OUT/train_90.csv" ] || [ ! -f "$EXP_OUT/val_10.csv" ]; then
    echo "[1a] split 90/10 (seed=$SEED, val_frac=$VAL_FRAC)"
    TRAIN_CSV="$TRAIN_CSV" EXP_OUT="$EXP_OUT" SEED="$SEED" VAL_FRAC="$VAL_FRAC" \
      python - <<'PY'
import os, pandas as pd, numpy as np
src = os.environ["TRAIN_CSV"]; out = os.environ["EXP_OUT"]
seed = int(os.environ["SEED"]); val_frac = float(os.environ["VAL_FRAC"])
df = pd.read_csv(src)
rng = np.random.RandomState(seed)
idx = rng.permutation(len(df))
n_val = max(1, int(len(df) * val_frac))
val_idx, tr_idx = idx[:n_val], idx[n_val:]
df.iloc[tr_idx].to_csv(f"{out}/train_90.csv", index=False)
df.iloc[val_idx].to_csv(f"{out}/val_10.csv", index=False)
print(f"[split] train={len(tr_idx)}  val={len(val_idx)}  src={src}")
PY
  else
    echo "[1a] reuse existing train_90.csv / val_10.csv"
  fi

  if [ ! -f "$EXP_OUT/best.pt" ]; then
    echo "[1b] train kcat-expert on iter_${it} (init from v6: $INIT_CKPT_V6)"
    [ -f "$INIT_CKPT_V6" ] || { echo "[ERR] INIT_CKPT_V6 not found: $INIT_CKPT_V6"; exit 1; }
    python "$ROOT/main_train_kcat_expert.py" \
      --train_csv "$EXP_OUT/train_90.csv" \
      --val_csv   "$EXP_OUT/val_10.csv" \
      --out_dir   "$EXP_OUT" \
      --esm_path  "$ESM_PATH" \
      --molt5_path "$MOLT5_PATH" \
      --rename_cols "$RENAME_JSON" \
      --batch_size "$EXPERT_BS" \
      --grad_accum "$EXPERT_GRAD_ACCUM" \
      --epochs    "$EXPERT_EPOCHS" \
      --lr "$EXPERT_LR" --lora_lr "$EXPERT_LORA_LR" \
      --weight_decay "$EXPERT_WD" \
      --esm_lora_rank "$EXPERT_LORA_RANK" --esm_lora_alpha "$EXPERT_LORA_ALPHA" \
      --rank_weight "$EXPERT_RANK_W" \
      --var_reg_weight "$EXPERT_VAR_W" \
      --nll_weight "$EXPERT_NLL_W" \
      --ccc_weight "$EXPERT_CCC_W" \
      --grad_clip "$EXPERT_GRAD_CLIP" \
      --randomize_smiles \
      --seq_crop_min "$EXPERT_SEQ_CROP" \
      --seq_mask_prob "$EXPERT_SEQ_MASK" \
      --patience "$EXPERT_PATIENCE" \
      --num_workers 4 \
      --log_every 50 \
      --init_ckpt "$INIT_CKPT_V6" \
      --seed "$SEED" 2>&1 | tee "$EXP_OUT/train.log"
  else
    echo "[1b] reuse existing best.pt"
  fi

  EMB_DIR="$EXP_OUT/moe_index"
  if [ ! -f "$EMB_DIR/train_emb.npy" ]; then
    echo "[1c] precompute train_emb.npy"
    python "$ROOT/tools/precompute_train_embed.py" \
      --train_csv "$EXP_OUT/train_90.csv" \
      --esm_path  "$ESM_PATH" \
      --out_dir   "$EMB_DIR" \
      --rename_cols "$RENAME_JSON" \
      --batch_size 8 --max_len 1024
  else
    echo "[1c] reuse existing train_emb.npy"
  fi
done

for it in "${INFER_ITERS[@]}"; do
  echo ""
  echo "##########################################################"
  echo "#  Step 2 / iter_${it}: MoE-soft inference"
  echo "##########################################################"

  if in_retrain "$it"; then
    EXPERT_CKPT_THIS="$EXPERT_OUT_ROOT/iter_${it}/best.pt"
    TRAIN_EMB_NPY_THIS="$EXPERT_OUT_ROOT/iter_${it}/moe_index/train_emb.npy"
    echo "[2.${it}] using retrained expert: $EXPERT_CKPT_THIS"
  else
    EXPERT_CKPT_THIS="$EXPERT_CKPT_PRESET"
    TRAIN_EMB_NPY_THIS="$TRAIN_EMB_NPY_PRESET"
    echo "[2.${it}] using PRESET expert:    $EXPERT_CKPT_THIS"
  fi

  EXPERT_CKPT="$EXPERT_CKPT_THIS" \
  TRAIN_EMB_NPY="$TRAIN_EMB_NPY_THIS" \
  OURS_WEIGHTS="$OURS_WEIGHTS" \
  OURS_SCALER="$OURS_SCALER" \
  ESM_PATH="$ESM_PATH" \
  MOLT5_PATH="$MOLT5_PATH" \
  DATA_ROOT="$DATA_ROOT_RO" \
  OUT_ROOT="$RESULTS_ROOT" \
  ITERS_TO_RUN="$it" \
  MODE=moe_softgate \
    bash "$ROOT/run_ours_10iter_inference.sh" 2>&1 | tee "$EXPERT_OUT_ROOT/iter_${it}_infer.log"
done

echo ""
echo "########## Step 3: aggregate leak-free results ##########"
RESULTS_ROOT="$RESULTS_ROOT" ITERS_TO_INFERENCE="$ITERS_TO_INFERENCE" \
python - <<'PY'
import os, numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

ROOT = os.environ["RESULTS_ROOT"]
iters = [f"iteration_{n}" for n in os.environ["ITERS_TO_INFERENCE"].split()]
splits = ["test", "testood99", "testood80", "testood60", "testood40"]
model = "ours-moe-soft"

rows = []
for it in iters:
    for sp in splits:
        p = f"{ROOT}/{it}/{sp}/{model}/predictions.csv"
        if not os.path.exists(p):
            print(f"[miss] {p}")
            continue
        df = pd.read_csv(p)
        yt = pd.to_numeric(df.get("y_true_kcat_log10"), errors="coerce").to_numpy()
        yp = pd.to_numeric(df.get("y_pred_kcat_log10"), errors="coerce").to_numpy()
        v = np.isfinite(yt) & np.isfinite(yp)
        if v.sum() < 2:
            continue
        yt, yp = yt[v], yp[v]
        r = float(pearsonr(yt, yp)[0]) if np.std(yt) > 0 and np.std(yp) > 0 else np.nan
        rows.append(dict(iteration=it, split=sp, model=model, n=int(v.sum()),
                         mae=float(mean_absolute_error(yt, yp)),
                         rmse=float(np.sqrt(mean_squared_error(yt, yp))),
                         r2=float(r2_score(yt, yp)),
                         pearson_r=r))

if not rows:
    print("[ERR] no predictions found"); raise SystemExit(1)

res = pd.DataFrame(rows)
os.makedirs(f"{ROOT}/summary", exist_ok=True)
res.to_csv(f"{ROOT}/summary/leakfree_by_iteration.csv", index=False)

agg = (res.groupby(["split","model"], as_index=False)
         .agg(n_iters=("n", lambda s: int((s>0).sum())),
              mae_mean=("mae","mean"), mae_std=("mae","std"),
              rmse_mean=("rmse","mean"), rmse_std=("rmse","std"),
              r2_mean=("r2","mean"), r2_std=("r2","std"),
              pearson_r_mean=("pearson_r","mean"),
              pearson_r_std=("pearson_r","std")))
agg.to_csv(f"{ROOT}/summary/leakfree_mean_std.csv", index=False)

print("\n========= Leak-free MoE-soft (kcat) =========")
print(agg.to_string(index=False))
print(f"\n[OK] {ROOT}/summary/leakfree_by_iteration.csv")
print(f"[OK] {ROOT}/summary/leakfree_mean_std.csv")
PY

echo ""
echo "[DONE]"
echo "  retrain iters:   ${ITERS_TO_RETRAIN}"
echo "  inference iters: ${ITERS_TO_INFERENCE}"
echo "  experts:         $EXPERT_OUT_ROOT/"
echo "  results:         $RESULTS_ROOT/"
echo "  summary:         $RESULTS_ROOT/summary/leakfree_mean_std.csv"
