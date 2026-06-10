#!/bin/bash
# Run TabSyn DP-SGD training and sampling for all datasets

set -e

source /home/lxler/miniconda3/bin/activate tame

DATASETS=("shoppers" "default" "cardio_train" "adult")
STEPS=3000
GPU=0
PHYSICAL_BATCH=64

for dataname in "${DATASETS[@]}"; do
    echo "========================================"
    echo "Dataset: $dataname"
    echo "========================================"

    # Train DP-SGD
    echo "[$(date)] Starting DP-SGD training..."
    python main.py --dataname "$dataname" --method tabsyn --mode train_sgd \
        --gpu "$GPU" --dp_steps "$STEPS" --dp_physical_batch_size "$PHYSICAL_BATCH"

    # Sample
    echo "[$(date)] Starting sampling..."
    python main.py --dataname "$dataname" --method tabsyn --mode sample \
        --sgd --save_path "sample_end_csv/tabsyn_sgd_${dataname}.csv" --gpu "$GPU"

    # Show metrics
    echo "[$(date)] DP metrics:"
    cat "tabsyn/ckpt_sgd/${dataname}/dp_metrics.json"

done

echo "========================================"
echo "All datasets completed!"
echo "========================================"
