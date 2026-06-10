import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")

from eval.eval_mle import eval_mle
from eval.eval_density import eval_density
from eval.eval_dcr import eval_dcr
from eval.eval_detection import eval_detection
from cal_memorization import cal_cat_ori, cal_mem_weight, cal_num_ori, cal_mem_ori

def eval_all(task_name, syn_data_path, dataname): 
    train_data = pd.read_csv(f"synthetic/{dataname}/real.csv")
    test_data = pd.read_csv(f"synthetic/{dataname}/test.csv")
    syn_data = pd.read_csv(syn_data_path)
    
    task_name = syn_data_path.split("/")[-1].split(".")[0]
    print(f"Start evaluating {task_name}...")
    
    # 打开要处理的文件
    with open(f"eval/result/{task_name}.txt", "a") as f:
        
        # 可选：测评memory
        print("start eval memory...")
        cat_mem, num_mem, mem_weight = cal_mem_weight(dataname, syn_data_path, train_data)
        f.write(f"mem_all: {mem_weight}, cat_ori: {cat_mem}, num_ori: {num_mem}\n")
        
        f.write("*********** Evaluation ***********\n\n")
        # 1. MLE
        print("start eval mle...")
        f.write(f"1. MLE: \n")
        overall_scores = eval_mle(syn_data.to_numpy(), test_data.to_numpy(), dataname)
        json.dump(overall_scores, f, indent=4, separators=(", ", ": "))
        
        # 2. density
        print("start eval density...")
        f.write(f"\n2. Density: \n")
        Shape, Trend, qual_report = eval_density(train_data, syn_data, dataname)
        f.write(f"Shape: {Shape}, Trend: {Trend}\n")
        
        # 3. dcr
        print("start eval dcr...")
        f.write(f"3. DCR: \n")
        score = eval_dcr(train_data, test_data, syn_data, dataname)
        f.write('DCR Score, a value closer to 0.5 is better\n')
        f.write(f'DCR Score = {score}\n')
        
        # 4. detection
        print("start eval detection...")
        f.write(f"4. Detection: \n")
        score = eval_detection(train_data, syn_data, dataname)
        f.write(f'Detection Score = {score}\n')
        
        # the end of this task, test the quality use bash
        print('start eval quality...')
        f.write('5. Quality: \n')
        # python eval/eval_quality.py ....
        
if __name__ == "__main__":
    # task_name = ""
    dataname = "shoppers" # TODO 改成对应的数据集
    
    syn_data_path_list = [
        # 'sample_end_csv/shopppers_sgd_dp_1.csv',
        # 'sample_end_csv/shopppers_sgd_dp_4.csv',
        'sample_end_csv/shoppers_sgd_dp_32.csv',
    ]
    
    for syn_data_path in syn_data_path_list:
        task_name = syn_data_path.split("/")[-1].split(".")[0]
        eval_all(task_name, syn_data_path, dataname=dataname)