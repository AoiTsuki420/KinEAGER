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
export OUT_ROOT=${OUT_ROOT:-/root/autodl-tmp/runs/core5_splits_revision_kcat}
export OURS_ENV=${OURS_ENV:-}
export MODE=${MODE:-single}
export OURS_WEIGHTS=${OURS_WEIGHTS:-/root/autodl-fs/runs/id_full_seed42/seed42/predictor_best_ema.pt}
export OURS_SCALER=${OURS_SCALER:-/root/autodl-fs/runs/id_full_seed42/seed42/scaler_fold.json}
export ESM_PATH=${ESM_PATH:-/root/autodl-fs/models/esm2_t33_650M_UR50D}
export MOLT5_PATH=${MOLT5_PATH:-/root/autodl-fs/models/molt5-base-smiles2caption}
export EXPERT_CKPT=${EXPERT_CKPT:-$ROOT/runs/kcat_expert_v7/best.pt}
export TRAIN_EMB_NPY=${TRAIN_EMB_NPY:-$ROOT/runs/moe_index/train_emb.npy}
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
unset TRANSFORMERS_OFFLINE HF_HUB_OFFLINE PIP_NO_INDEX 2>/dev/null || true

__check_writable () {
  local d="$1"
  mkdir -p "$d" 2>/dev/null && [ -w "$d" ]
}
DEFAULT_HF_HUB="$HOME/.cache/huggingface/hub"
DEFAULT_HF_HOME="$HOME/.cache/huggingface"
DEFAULT_TORCH="$HOME/.cache/torch"
if [ -n "${TRANSFORMERS_CACHE:-}" ] && ! __check_writable "$TRANSFORMERS_CACHE"; then
  echo "[warn] TRANSFORMERS_CACHE='$TRANSFORMERS_CACHE' not writable, fallback to $DEFAULT_HF_HUB"
  unset TRANSFORMERS_CACHE
fi
if [ -n "${HF_HOME:-}" ] && ! __check_writable "$HF_HOME"; then
  echo "[warn] HF_HOME='$HF_HOME' not writable, fallback to $DEFAULT_HF_HOME"
  unset HF_HOME
fi
if [ -n "${TORCH_HOME:-}" ] && ! __check_writable "$TORCH_HOME"; then
  echo "[warn] TORCH_HOME='$TORCH_HOME' not writable, fallback to $DEFAULT_TORCH"
  unset TORCH_HOME
fi
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$DEFAULT_HF_HUB}
export HF_HOME=${HF_HOME:-$DEFAULT_HF_HOME}
export TORCH_HOME=${TORCH_HOME:-$DEFAULT_TORCH}

mkdir -p "$OUT_ROOT/_logs" "$OUT_ROOT/summary" "$TRANSFORMERS_CACHE" "$TORCH_HOME/hub/checkpoints"

ESM2_LOCAL_PT=${ESM2_LOCAL_PT:-/root/autodl-fs/models/esm2_t33_650M_UR50D/esm2_t33_650M_UR50D.pt}
[ -f "$ESM2_LOCAL_PT" ] && cp -f "$ESM2_LOCAL_PT" "$TORCH_HOME/hub/checkpoints/esm2_t33_650M_UR50D.pt" || true

preflight () {
  local PY_CMD
  if [ -n "$OURS_ENV" ]; then
    PY_CMD="conda run --no-capture-output -n $OURS_ENV python"
  else
    PY_CMD="python"
  fi

  if ! $PY_CMD -c "import transformers, torch, sklearn, scipy, pandas" 2>/dev/null; then
    echo "[ERR] preflight: missing transformers/torch/sklearn/scipy/pandas in env '${OURS_ENV:-current shell}'"
    echo "      conda activate <your-env>; bash $(basename "$0")"
    exit 1
  fi

  for pth in "$ESM_PATH" "$MOLT5_PATH"; do
    if [ ! -d "$pth" ] || [ ! -f "$pth/config.json" ]; then
      echo "[ERR] preflight: model dir not found or invalid: $pth"
      echo "      请确认本地有 ESM-650M 与 MolT5 解压目录, 或通过环境变量覆盖:"
      echo "        ESM_PATH=/path/to/esm2_t33_650M_UR50D bash $(basename "$0")"
      echo "        MOLT5_PATH=/path/to/molt5-base-smiles2caption bash $(basename "$0")"
      exit 1
    fi
  done

  if [ ! -f "$OURS_WEIGHTS" ]; then
    echo "[ERR] preflight: main ckpt missing: $OURS_WEIGHTS"; exit 1
  fi

  if [ "$MODE" = "ensemble" ]; then
    [ -f "$EXPERT_CKPT" ] || { echo "[ERR] ensemble mode: EXPERT_CKPT missing: $EXPERT_CKPT"; exit 1; }
    [ -f "$TRAIN_EMB_NPY" ] || { echo "[ERR] ensemble mode: TRAIN_EMB_NPY missing: $TRAIN_EMB_NPY"; exit 1; }
    echo "[WARN] ensemble mode 选用了, 但 splits_revision csv 通常不带 prot_pdb_path/lig_sdf_path,"
    echo "       MoE 会因 has_struct=False 全程硬门控回 main, 实际等同 single 但慢 ~2x. 确认要继续?"
    sleep 3
  fi

  echo "[OK] preflight: env='${OURS_ENV:-current shell}' mode=$MODE"
  echo "     ESM    = $ESM_PATH"
  echo "     MolT5  = $MOLT5_PATH"
  echo "     main   = $OURS_WEIGHTS"
  if [ "$MODE" = "ensemble" ]; then
    echo "     expert = $EXPERT_CKPT"
    echo "     index  = $TRAIN_EMB_NPY"
  fi
}
preflight

pick_csv () {
  local iter="$1" split="$2"
  case "$split" in
    test)      ls "$DATA_ROOT/$iter"/kcat_test_split_*.csv 2>/dev/null | head -n1 ;;
    testood99) echo "$DATA_ROOT/$iter/kcat-seq_test_sequence_99cluster.csv" ;;
    testood80) echo "$DATA_ROOT/$iter/kcat-seq_test_sequence_80cluster.csv" ;;
    testood60) echo "$DATA_ROOT/$iter/kcat-seq_test_sequence_60cluster.csv" ;;
    testood40) echo "$DATA_ROOT/$iter/kcat-seq_test_sequence_40cluster.csv" ;;
    *) return 1 ;;
  esac
}

run_py () {
  if [ -n "$OURS_ENV" ]; then
    conda run --no-capture-output -n "$OURS_ENV" python "$@"
  else
    python "$@"
  fi
}

run_inline_py () {
  if [ -n "$OURS_ENV" ]; then
    conda run --no-capture-output -n "$OURS_ENV" python -
  else
    python -
  fi
}

if [ -n "${ITERS_TO_RUN:-}" ]; then
  ITER_LIST=()
  for n in $ITERS_TO_RUN; do ITER_LIST+=("iteration_$n"); done
else
  ITER_LIST=(iteration_{1..10})
fi
echo "===== Phase 1: per-iteration inference (mode=$MODE) ====="
echo "       iters: ${ITER_LIST[*]}"
for ITER in "${ITER_LIST[@]}"; do
  for SPLIT in test testood99 testood80 testood60 testood40; do
    TEST_CSV=$(pick_csv "$ITER" "$SPLIT")
    [ -f "$TEST_CSV" ] || { echo "[MISS] $TEST_CSV — skip"; continue; }

    case "$MODE" in
      single)        SUB=ours ;;
      ensemble)      SUB=ours-moe-hard ;;
      moe_softgate)  SUB=ours-moe-soft ;;
      *) echo "[ERR] unknown MODE=$MODE"; exit 1 ;;
    esac
    OUT_DIR="$OUT_ROOT/$ITER/$SPLIT/$SUB"
    mkdir -p "$OUT_DIR"

    if [ -s "$OUT_DIR/predictions.csv" ]; then
      echo "[SKIP] $ITER $SPLIT $SUB (predictions.csv exists)"
      continue
    fi

    LOG="$OUT_ROOT/_logs/${SUB}_${ITER}_${SPLIT}.log"
    echo "[RUN ] $ITER $SPLIT $SUB  ->  $OUT_DIR"

    cd "$OUT_DIR"

    if [ "$MODE" = "single" ]; then
      run_py "$ROOT/main_infer_predictor.py" \
        -csv "$TEST_CSV" \
        -weights "$OURS_WEIGHTS" \
        -device cuda \
        -esm "$ESM_PATH" \
        -molt5 "$MOLT5_PATH" \
        -scaler "$OURS_SCALER" 2>&1 | tee "$LOG"

      TEST_CSV="$TEST_CSV" OUT_DIR="$OUT_DIR" run_inline_py <<'PY'
import os, numpy as np, pandas as pd
base = pd.read_csv(os.environ["TEST_CSV"]).reset_index(drop=True)
raw_path = f"{os.environ['OUT_DIR']}/predictions_kcat.csv"
if not os.path.exists(raw_path):  # 兼容老路径
    alt = f"{os.environ['OUT_DIR']}/predictions.csv"
    if os.path.exists(alt):
        raw_path = alt
raw  = pd.read_csv(raw_path)
cm   = {c.lower(): c for c in base.columns}
rid  = cm.get("row_id")
src  = cm.get("source_id") or cm.get("sequence_source") or cm.get("group")
ylog = cm.get("log10kcat_max") or cm.get("log10_value")
ylin = cm.get("kcat(s^-1)") or cm.get("kcat") or cm.get("value")
if rid is None:
    base["row_id"] = [f"row_{i}" for i in range(len(base))]; rid = "row_id"
if ylog is not None:
    y_true = pd.to_numeric(base[ylog], errors="coerce")
elif ylin is not None:
    y_true = np.log10(pd.to_numeric(base[ylin], errors="coerce").clip(lower=1e-12))
else:
    y_true = pd.Series([np.nan]*len(base))
if "pred_kcat_log" in raw.columns:
    pred_log = pd.to_numeric(raw["pred_kcat_log"], errors="coerce")
elif "pred_kcat" in raw.columns:
    pred_lin = pd.to_numeric(raw["pred_kcat"], errors="coerce")
    pred_log = np.log10(pred_lin.clip(lower=1e-12))
else:
    pred_log = pd.Series([np.nan]*len(base))
if "unc_kcat_log" in raw.columns:
    sigma = pd.to_numeric(raw["unc_kcat_log"], errors="coerce")
elif "sigma_kcat" in raw.columns:
    sigma = pd.to_numeric(raw["sigma_kcat"], errors="coerce")
else:
    sigma = pd.Series([np.nan]*len(base))
n = min(len(base), len(pred_log))
pd.DataFrame({
    "row_id":            base[rid].astype(str).iloc[:n],
    "source_id":         (base[src].astype(str).iloc[:n] if src else "unknown"),
    "y_true_kcat_log10": pd.Series(y_true).iloc[:n],
    "y_pred_kcat_log10": pd.Series(pred_log).iloc[:n],
    "sigma_kcat_log10":  pd.Series(sigma).iloc[:n],
}).to_csv(f"{os.environ['OUT_DIR']}/predictions.csv", index=False)
PY

    elif [ "$MODE" = "ensemble" ] || [ "$MODE" = "moe_softgate" ]; then
      RAW_OUT="$OUT_DIR/moe_raw.csv"
      RENAME_JSON='{"reactant_smiles":"smiles","log10kcat_max":"kcat_log10"}'
      EXTRA=()
      if [ "$MODE" = "moe_softgate" ]; then
        EXTRA+=(--no_hard_gate)
      fi
      run_py "$ROOT/main_infer_ensemble.py" \
        --test_csv "$TEST_CSV" \
        --main_ckpt "$OURS_WEIGHTS" \
        --expert_ckpt "$EXPERT_CKPT" \
        --train_emb_npy "$TRAIN_EMB_NPY" \
        --out_csv "$RAW_OUT" \
        --esm_path "$ESM_PATH" \
        --molt5_path "$MOLT5_PATH" \
        --main_esm "$ESM_PATH" \
        --main_molt5 "$MOLT5_PATH" \
        --rename_cols "$RENAME_JSON" \
        --mc_samples 5 --batch_size 16 \
        "${EXTRA[@]}" 2>&1 | tee "$LOG"

      TEST_CSV="$TEST_CSV" OUT_DIR="$OUT_DIR" RAW_OUT="$RAW_OUT" run_inline_py <<'PY'
import os, numpy as np, pandas as pd
base = pd.read_csv(os.environ["TEST_CSV"]).reset_index(drop=True)
raw  = pd.read_csv(os.environ["RAW_OUT"])
cm   = {c.lower(): c for c in base.columns}
rid  = cm.get("row_id")
src  = cm.get("source_id") or cm.get("sequence_source") or cm.get("group")
ylog = cm.get("log10kcat_max") or cm.get("log10_value")
ylin = cm.get("kcat(s^-1)") or cm.get("kcat") or cm.get("value")
if rid is None:
    base["row_id"] = [f"row_{i}" for i in range(len(base))]; rid = "row_id"
if ylog is not None:
    y_true = pd.to_numeric(base[ylog], errors="coerce")
elif ylin is not None:
    y_true = np.log10(pd.to_numeric(base[ylin], errors="coerce").clip(lower=1e-12))
else:
    y_true = pd.Series([np.nan]*len(base))
pred_log = pd.to_numeric(raw.get("mu_ensemble"), errors="coerce")
sigma    = pd.to_numeric(raw.get("s2_ensemble"), errors="coerce").clip(lower=0).pow(0.5)
n = min(len(base), len(pred_log))
pd.DataFrame({
    "row_id":            base[rid].astype(str).iloc[:n],
    "source_id":         (base[src].astype(str).iloc[:n] if src else "unknown"),
    "y_true_kcat_log10": pd.Series(y_true).iloc[:n],
    "y_pred_kcat_log10": pd.Series(pred_log).iloc[:n],
    "sigma_kcat_log10":  pd.Series(sigma).iloc[:n],
}).to_csv(f"{os.environ['OUT_DIR']}/predictions.csv", index=False)
PY

    else
      echo "[ERR] unknown MODE=$MODE"; exit 1
    fi
  done
done

echo "===== Phase 2: summarize ====="
OUT_ROOT="$OUT_ROOT" run_inline_py <<'PY'
import os, numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

OUT = os.environ["OUT_ROOT"]
iters  = [f"iteration_{i}" for i in range(1, 11)]
splits = ["test", "testood99", "testood80", "testood60", "testood40"]
models = ["ours", "ours-moe-hard", "ours-moe-soft",
          "catapro", "catpred", "unikp", "dlkcat"]

rows = []
for it in iters:
    for sp in splits:
        for md in models:
            p = f"{OUT}/{it}/{sp}/{md}/predictions.csv"
            if not os.path.exists(p):
                rows.append(dict(iteration=it, split=sp, model=md, kcat_n=0,
                                 kcat_mae=np.nan, kcat_rmse=np.nan,
                                 kcat_r2=np.nan, kcat_pearson_r=np.nan, coverage=0.0))
                continue
            df = pd.read_csv(p)
            yt = pd.to_numeric(df.get("y_true_kcat_log10"), errors="coerce").to_numpy()
            yp = pd.to_numeric(df.get("y_pred_kcat_log10"), errors="coerce").to_numpy()
            v  = np.isfinite(yt) & np.isfinite(yp)
            cov = float(v.mean()) if len(v) else 0.0
            if v.sum() < 2:
                rows.append(dict(iteration=it, split=sp, model=md, kcat_n=int(v.sum()),
                                 kcat_mae=np.nan, kcat_rmse=np.nan,
                                 kcat_r2=np.nan, kcat_pearson_r=np.nan, coverage=cov))
                continue
            yt, yp = yt[v], yp[v]
            r, _ = pearsonr(yt, yp) if np.std(yt) > 0 and np.std(yp) > 0 else (np.nan, None)
            rows.append(dict(iteration=it, split=sp, model=md, kcat_n=int(v.sum()),
                             kcat_mae=float(mean_absolute_error(yt, yp)),
                             kcat_rmse=float(np.sqrt(mean_squared_error(yt, yp))),
                             kcat_r2=float(r2_score(yt, yp)),
                             kcat_pearson_r=float(r), coverage=cov))

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/summary/core5_metrics_by_iteration.csv", index=False)
agg = (res.groupby(["split","model"], as_index=False)
         .agg(files_found=("kcat_n", lambda s: int((s>0).sum())),
              kcat_n_mean=("kcat_n","mean"),
              coverage_mean=("coverage","mean"),
              mae_mean=("kcat_mae","mean"),  mae_std=("kcat_mae","std"),
              rmse_mean=("kcat_rmse","mean"), rmse_std=("kcat_rmse","std"),
              r2_mean=("kcat_r2","mean"),    r2_std=("kcat_r2","std"),
              pearson_r_mean=("kcat_pearson_r","mean"),
              pearson_r_std=("kcat_pearson_r","std")))
agg.to_csv(f"{OUT}/summary/core5_metrics_mean_std.csv", index=False)

ours_models = ["ours", "ours-moe-hard", "ours-moe-soft"]
ours = res[res.model.isin(ours_models)].copy()
ours.to_csv(f"{OUT}/summary/ours_kcat_by_iteration.csv", index=False)
ours_agg = agg[agg.model.isin(ours_models)].copy()
ours_agg.to_csv(f"{OUT}/summary/ours_kcat_mean_std.csv", index=False)

print("\n========= Ours mean ± std (kcat) =========")
print(ours_agg.to_string(index=False))
print("\n========= All models mean ± std (kcat) =========")
print(agg.sort_values(["split","mae_mean"]).to_string(index=False))
print(f"\n[OK] {OUT}/summary/ours_kcat_mean_std.csv")
print(f"[OK] {OUT}/summary/core5_metrics_mean_std.csv")
PY

echo "===== Done ====="
