#!/usr/bin/env bash
set -euo pipefail

CSV=/root/autodl-fs/data/unified_train_mixed_phys_norm.csv
NPZ=/root/autodl-fs/data/unified_train_mixed_struct.npz
ESM=/root/autodl-fs/models/esm2_t33_650M_UR50D
MOLT5=/root/autodl-fs/models/molt5-base-smiles2caption
RUNS=/root/autodl-fs/runs
SCRIPT=main_train_predictor_multigpu.py     # 位于项目根

declare -A EXP_FULL=( [42]=0.8195 [43]=0.8177 [44]=0.8130 )
EXP_SUB_S42=0.5549

for SEED in 42 43 44; do
  WEIGHTS=${RUNS}/id_full_seed${SEED}_ddp/seed${SEED}/predictor_best_ema.pt
  OUT_RUN=id_full_seed${SEED}_evalsub          # 独立输出目录，避免覆盖原结果
  OUT_DIR=${RUNS}/${OUT_RUN}/seed${SEED}

  echo "=================================================================="
  echo "[SEED ${SEED}] weights = ${WEIGHTS}"
  echo "[SEED ${SEED}] 期望 full metrics.json macro_mae ≈ ${EXP_FULL[$SEED]}"
  if [ "${SEED}" = "42" ]; then
    echo "[SEED 42] 期望 struct-subset macro_mae ≈ ${EXP_SUB_S42}（验证闸门）"
  fi
  echo "=================================================================="

  if [ ! -f "${WEIGHTS}" ]; then
    echo "[ERROR] 权重不存在：${WEIGHTS}（请确认 run_dir 路径）"; exit 1
  fi

  python ${SCRIPT} \
    -csv ${CSV} \
    --out_dir ${RUNS} \
    --run_name ${OUT_RUN} \
    --seed ${SEED} \
    --eval_only \
    --weights ${WEIGHTS} \
    -esm ${ESM} \
    -molt5 ${MOLT5} \
    -batch_size 24 \
    -device cuda \
    --use_interactions --use_gate_p --use_gate_l \
    --use_struct_branch --struct_in_dim_prot 45 --struct_in_dim_lig 135 --struct_fusion_mode concat \
    --final_contact_npz ${NPZ} --struct_ablation_mode real --struct_random_seed 123 \
    --source_col source_id --default_source_id 0 \
    --enable_avail_gate --avail_gate_hidden 256 \
    --domain_uncertainty --num_domains 3 \
    --use_phys_evidence \
    --split_mode group_pair --split_seed 42 --val_ratio 0.2 --test_ratio 0.2 --ood_dedup_pair \
    --test_struct_keep_ratio 1.0

  echo ""
  echo "[SEED ${SEED}] ---- 完整测试集 metrics.json ----"
  python - "$OUT_DIR" <<'PY'
import json, sys, os
d = sys.argv[1]
m = json.load(open(os.path.join(d, "metrics.json")))
print(f"  macro_mae={m['macro_mae']:.6f}  macro_r2={m['macro_r2']:.6f}  macro_nll={m['macro_nll']:.6f}")
print(f"  n: kcat={m['kcat_n']}  Km={m['Km_n']}  kcat/Km={m['kcat/Km_n']}")
PY

  echo "[SEED ${SEED}] ---- 结构可用子集 metrics_struct_available.json ----"
  python - "$OUT_DIR" <<'PY'
import json, sys, os
d = sys.argv[1]
m = json.load(open(os.path.join(d, "metrics_struct_available.json")))
print(f"  macro_mae={m['macro_mae']:.6f}  macro_r2={m['macro_r2']:.6f}  macro_nll={m['macro_nll']:.6f}")
print(f"  macro_cov90={m['macro_cov90']:.4f}  macro_cov95={m['macro_cov95']:.4f}")
print(f"  n: kcat={m['kcat_n']}  Km={m['Km_n']}  kcat/Km={m['kcat/Km_n']}  (期望 4438 / 4850 / 3087)")
PY
  echo ""
done

echo "=================================================================="
echo "全部完成。请先核对 seed42：full≈0.8195、subset≈0.5549、n=4438/4850/3087。"
echo "若 seed42 对齐，则 seed43/44 的 metrics_struct_available.json 即可写入 Table 2。"
echo "把三个 *_evalsub/seed*/metrics_struct_available.json 回传本地即可。"
echo "=================================================================="
