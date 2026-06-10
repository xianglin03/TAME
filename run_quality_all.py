import sys
sys.path.insert(0, '/home/lxler/work/TAME')

from eval.eval_quality import eval_quality
import pandas as pd

DATASETS = {
    "shoppers": ("sample_end_csv/tabsyn_sgd_shoppers.csv", "synthetic/shoppers/real.csv"),
    "default": ("sample_end_csv/tabsyn_sgd_default.csv", "synthetic/default/real.csv"),
    "cardio_train": ("sample_end_csv/tabsyn_sgd_cardio_train.csv", "synthetic/cardio_train/real.csv"),
    "adult": ("sample_end_csv/tabsyn_sgd_adult.csv", "synthetic/adult/real.csv"),
}

results = {}
for dataname, (syn_path, real_path) in DATASETS.items():
    print(f"\n{'='*50}")
    print(f"Quality eval: {dataname}")
    print(f"{'='*50}")
    try:
        train_data = pd.read_csv(real_path)
        syn_data = pd.read_csv(syn_path)
        alpha, beta = eval_quality(train_data, syn_data, dataname, "tabsyn")
        results[dataname] = (alpha, beta)
        print(f"Alpha: {alpha:.4f}, Beta: {beta:.4f}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*50)
print("Summary:")
for dataname, (alpha, beta) in results.items():
    print(f"{dataname}: Alpha={alpha:.4f}, Beta={beta:.4f}")
