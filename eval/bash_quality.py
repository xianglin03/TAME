import argparse
import pandas as pd
import sys

# sys.path.append("./eval")

from eval.eval_quality import eval_quality

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default=None)

args = parser.parse_args()

task_name = args.task_name

train_data = pd.read_csv("synthetic/shoppers/real.csv")
syn_data = pd.read_csv(f"sample_end_csv/{task_name}.csv")

alpha, beta = eval_quality(train_data, syn_data, "shoppers", "tabddpm")

with open(f"eval/result/{task_name}.txt", "a") as f:
    f.write(f"Alpha: {alpha}, Beta: {beta}\n")

print("Evaluation completed!!!")