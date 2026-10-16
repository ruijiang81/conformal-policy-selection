#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/experiments:${PYTHONPATH}"
DATA_PATH="${DIABETES_DATA_PATH:-}"

# --- Synthetic: Fig 1 (both correct), Fig 2 + Table 1 (four DR scenarios) ---
python experiments/doubly_robust.py \
    --n_train 200 --n_calib 500 --n_test 100 --p 5 --sig 1.0 --c 0.0 \
    --n_reps 400 --seed 42 --methods BH hete \
    --output_dir results/doubly_robust

# --- SI: LaLonde job training (threshold c=4000) ---
python experiments/lalonde.py \
    --data_path data/lalonde_data.csv --c 4000 --n_reps 100 --seed 42 \
    --methods BH hete --output_dir results/lalonde

# --- SI Sec 4: baseline comparison under covariate shift (Table/Fig S3) ---
python experiments/baseline_comparison.py \
    --n_reps 30 --seed 42 --output_dir results/baseline_comparison

if [ -z "$DATA_PATH" ]; then
    echo "DIABETES_DATA_PATH is not set; skipping the diabetes (EHR) experiments."
    exit 0
fi

# --- Fig 3: diabetes safety gate (requires EHR data under DUA) ---
python experiments/diabetes_safety_gate.py \
    --data_path "$DATA_PATH" --n_reps 20 --output_dir results/diabetes_safety_gate

# --- SI Sec 5: heterogeneous safety gate (Table/Fig S4; requires EHR data) ---
python experiments/diabetes_safety_gate_hetero.py \
    --data_path "$DATA_PATH" --n_reps 20 --max_test 250 \
    --mu0_spread 0.18 --tau_scale 0.5 --sigma 0.04 --null_quantile 0.32 \
    --pos_frac 0.5 --cate_learner t --behavior_scale 1.5 \
    --output_dir results/diabetes_safety_gate_hetero
