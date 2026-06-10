import sys
sys.path.insert(0, '/home/lxler/work/TAME')
from eval.eval_all import eval_all

DATASETS = {
    "shoppers": "sample_end_csv/tabsyn_sgd_shoppers.csv",
    "default": "sample_end_csv/tabsyn_sgd_default.csv",
    "cardio_train": "sample_end_csv/tabsyn_sgd_cardio_train.csv",
    "adult": "sample_end_csv/tabsyn_sgd_adult.csv",
}

for dataname, syn_path in DATASETS.items():
    task_name = syn_path.split("/")[-1].split(".")[0]
    print(f"\n{'='*50}")
    print(f"Evaluating {dataname}: {task_name}")
    print(f"{'='*50}")
    try:
        eval_all(task_name, syn_path, dataname)
        print(f"Done: eval/result/{task_name}.txt")
    except Exception as e:
        print(f"Error evaluating {dataname}: {e}")
        import traceback
        traceback.print_exc()
