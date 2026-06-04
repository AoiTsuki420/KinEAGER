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
export OUT_ROOT=${OUT_ROOT:-/root/autodl-tmp/runs/core5_splits_revision_km}
export OURS_ENV=${OURS_ENV:-}
export OURS_WEIGHTS=${OURS_WEIGHTS:-/root/autodl-fs/runs/id_full_seed42/seed42/predictor_best_ema.pt}
export OURS_SCALER=${OURS_SCALER:-/root/autodl-fs/runs/id_full_seed42/seed42/scaler_fold.json}
export ESM_PATH=${ESM_PATH:-/root/autodl-fs/models/esm2_t33_650M_UR50D}
export MOLT5_PATH=${MOLT5_PATH:-/root/autodl-fs/models/molt5-base-smiles2caption}
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
  unset TRANSFORMERS_CACHE
fi
if [ -n "${HF_HOME:-}" ] && ! __check_writable "$HF_HOME"; then
  unset HF_HOME
fi
if [ -n "${TORCH_HOME:-}" ] && ! __check_writable "$TORCH_HOME"; then
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
    echo "[ERR] preflight: missing transformers/torch/sklearn/scipy/pandas"
    exit 1
  fi
  for pth in "$ESM_PATH" "$MOLT5_PATH"; do
    if [ ! -d "$pth" ] || [ ! -f "$pth/config.json" ]; then
      echo "[ERR] preflight: model dir invalid: $pth"; exit 1
    fi
  done
  [ -f "$OURS_WEIGHTS" ] || { echo "[ERR] missing main ckpt: $OURS_WEIGHTS"; exit 1; }
  echo "[OK] preflight (Km task): ESM=$ESM_PATH | MolT5=$MOLT5_PATH | main=$OURS_WEIGHTS"
}
preflight

pick_csv () {
  local iter="$1" split="$2"
  case "$split" in
    test)      ls "$DATA_ROOT/$iter"/km_test_split_*.csv 2>/dev/null | head -n1 ;;
    testood99) echo "$DATA_ROOT/$iter/km-seq_test_sequence_99cluster.csv" ;;
    testood80) echo "$DATA_ROOT/$iter/km-seq_test_sequence_80cluster.csv" ;;
    testood60) echo "$DATA_ROOT/$iter/km-seq_test_sequence_60cluster.csv" ;;
    testood40) echo "$DATA_ROOT/$iter/km-seq_test_sequence_40cluster.csv" ;;
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
echo "===== Phase 1: per-iteration Km inference ====="
echo "       iters: ${ITER_LIST[*]}"
for ITER in "${ITER_LIST[@]}"; do
  for SPLIT in test testood99 testood80 testood60 testood40; do
    TEST_CSV=$(pick_csv "$ITER" "$SPLIT")
    [ -f "$TEST_CSV" ] || { echo "[MISS] $TEST_CSV — skip"; continue; }

    SUB=ours
    OUT_DIR="$OUT_ROOT/$ITER/$SPLIT/$SUB"
    mkdir -p "$OUT_DIR"

    if [ -s "$OUT_DIR/predictions.csv" ]; then
      echo "[SKIP] $ITER $SPLIT $SUB (predictions.csv exists)"
      continue
    fi

    LOG="$OUT_ROOT/_logs/${SUB}_${ITER}_${SPLIT}.log"
    cd "$OUT_DIR"

    if [ -s "$OUT_DIR/predictions_km.csv" ] || [ -s "$OUT_DIR/predictions_kcat.csv" ]; then
      echo "[XW  ] $ITER $SPLIT $SUB  -> reuse existing raw predictions"
    else
      echo "[RUN ] $ITER $SPLIT $SUB  ->  $OUT_DIR"
      run_py "$ROOT/main_infer_predictor.py" \
        -csv "$TEST_CSV" \
        -weights "$OURS_WEIGHTS" \
        -device cuda \
        -esm "$ESM_PATH" \
        -molt5 "$MOLT5_PATH" \
        -scaler "$OURS_SCALER" \
        --task km 2>&1 | tee "$LOG"
    fi

    TEST_CSV="$TEST_CSV" OUT_DIR="$OUT_DIR" run_inline_py <<'PY'
import os, numpy as np, pandas as pd
base = pd.read_csv(os.environ["TEST_CSV"]).reset_index(drop=True)
raw_path = None
for cand in ("predictions_km.csv", "predictions_kcat.csv", "predictions.csv"):
    p = f"{os.environ['OUT_DIR']}/{cand}"
    if os.path.exists(p):
        raw_path = p; break
if raw_path is None:
    raise FileNotFoundError(f"no predictions_*.csv under {os.environ['OUT_DIR']}")
raw = pd.read_csv(raw_path)
cm  = {c.lower(): c for c in base.columns}
rid = cm.get("row_id")
src = cm.get("source_id") or cm.get("sequence_source") or cm.get("group")
ylog_col = cm.get("log10_value")
ylin_col = cm.get("km(m)") or cm.get("km") or cm.get("km_value") or cm.get("value")
if rid is None:
    base["row_id"] = [f"row_{i}" for i in range(len(base))]; rid = "row_id"
if ylog_col is not None:
    y_true_M = pd.to_numeric(base[ylog_col], errors="coerce")
    y_true_mM = y_true_M + 3.0    # log10(M) → log10(mM)
elif ylin_col is not None:
    y_true_lin_M = pd.to_numeric(base[ylin_col], errors="coerce").clip(lower=1e-12)
    y_true_mM = np.log10(y_true_lin_M) + 3.0
else:
    y_true_mM = pd.Series([np.nan]*len(base))
if "pred_Km_log" in raw.columns:
    pred_log = pd.to_numeric(raw["pred_Km_log"], errors="coerce")
elif "pred_km_log" in raw.columns:
    pred_log = pd.to_numeric(raw["pred_km_log"], errors="coerce")
else:
    pred_log = pd.Series([np.nan]*len(raw))
sigma_col = next((c for c in raw.columns if c.lower() in {"unc_km_log","unc_km","sigma_km","sigma_km_log"}), None)
sigma = pd.to_numeric(raw[sigma_col], errors="coerce") if sigma_col else pd.Series([np.nan]*len(raw))
n = min(len(base), len(pred_log))
pd.DataFrame({
    "row_id":           base[rid].astype(str).iloc[:n],
    "source_id":        (base[src].astype(str).iloc[:n] if src else "unknown"),
    "y_true_km_log10":  pd.Series(y_true_mM).iloc[:n],
    "y_pred_km_log10":  pd.Series(pred_log).iloc[:n],
    "sigma_km_log10":   pd.Series(sigma).iloc[:n],
}).to_csv(f"{os.environ['OUT_DIR']}/predictions.csv", index=False)
PY
  done
done

echo "===== Phase 2: summarize (Km) ====="
OUT_ROOT="$OUT_ROOT" run_inline_py <<'PY'
import os, numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

OUT = os.environ["OUT_ROOT"]
iters  = [f"iteration_{i}" for i in range(1, 11)]
splits = ["test", "testood99", "testood80", "testood60", "testood40"]
models = ["ours", "catapro_km"]   # 与 catapro_km_by_iteration_with_coverage.csv 同 schema

rows = []
for it in iters:
    for sp in splits:
        for md in models:
            p = f"{OUT}/{it}/{sp}/{md}/predictions.csv"
            if not os.path.exists(p):
                rows.append(dict(iteration=it, split=sp, model=md, km_n=0,
                                 km_mae=np.nan, km_rmse=np.nan,
                                 km_r2=np.nan, km_pearson_r=np.nan, coverage=0.0))
                continue
            df = pd.read_csv(p)
            yt = pd.to_numeric(df.get("y_true_km_log10"), errors="coerce").to_numpy()
            yp = pd.to_numeric(df.get("y_pred_km_log10"), errors="coerce").to_numpy()
            v  = np.isfinite(yt) & np.isfinite(yp)
            cov = float(v.mean()) if len(v) else 0.0
            if v.sum() < 2:
                rows.append(dict(iteration=it, split=sp, model=md, km_n=int(v.sum()),
                                 km_mae=np.nan, km_rmse=np.nan,
                                 km_r2=np.nan, km_pearson_r=np.nan, coverage=cov))
                continue
            yt, yp = yt[v], yp[v]
            r, _ = pearsonr(yt, yp) if np.std(yt) > 0 and np.std(yp) > 0 else (np.nan, None)
            rows.append(dict(iteration=it, split=sp, model=md, km_n=int(v.sum()),
                             km_mae=float(mean_absolute_error(yt, yp)),
                             km_rmse=float(np.sqrt(mean_squared_error(yt, yp))),
                             km_r2=float(r2_score(yt, yp)),
                             km_pearson_r=float(r), coverage=cov))

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/summary/km_metrics_by_iteration.csv", index=False)
agg = (res.groupby(["split","model"], as_index=False)
         .agg(files_found=("km_n", lambda s: int((s>0).sum())),
              km_n_mean=("km_n","mean"),
              coverage_mean=("coverage","mean"),
              mae_mean=("km_mae","mean"),  mae_std=("km_mae","std"),
              rmse_mean=("km_rmse","mean"), rmse_std=("km_rmse","std"),
              r2_mean=("km_r2","mean"),    r2_std=("km_r2","std"),
              pearson_r_mean=("km_pearson_r","mean"),
              pearson_r_std=("km_pearson_r","std")))
agg.to_csv(f"{OUT}/summary/km_metrics_mean_std.csv", index=False)

ours = res[res.model == "ours"].copy()
ours.to_csv(f"{OUT}/summary/ours_km_by_iteration.csv", index=False)
ours_agg = agg[agg.model == "ours"].copy()
ours_agg.to_csv(f"{OUT}/summary/ours_km_mean_std.csv", index=False)

print("\n========= Ours mean ± std (Km) =========")
print(ours_agg.to_string(index=False))
print("\n========= All models mean ± std (Km) =========")
print(agg.sort_values(["split","mae_mean"]).to_string(index=False))
print(f"\n[OK] {OUT}/summary/ours_km_mean_std.csv")
print(f"[OK] {OUT}/summary/km_metrics_mean_std.csv")
PY

echo "===== Done ====="
