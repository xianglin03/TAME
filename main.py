import torch
from utils import execute_function, get_args

if __name__ == '__main__':
    args = get_args()
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if args.gpu < 0 or args.gpu >= gpu_count:
            print(f"Requested --gpu={args.gpu}, but only {gpu_count} CUDA device(s) are available. Fallback to --gpu=0.")
            args.gpu = 0
        args.device = f'cuda:{args.gpu}'
    else:
        args.gpu = -1
        args.device = 'cpu'

    if not args.save_path:
        args.save_path = f'synthetic/{args.dataname}/{args.method}.csv'
        
    main_fn = execute_function(args.method, args.mode)

    main_fn(args)

# tabddpm
# python main.py --dataname shoppers --method tabddpm --mode train                                                                      # train model
# python main.py --dataname shoppers --method tabddpm --mode sample --save_path sample_end_csv/lxler_test.csv --task_name lxler_test    # sample from model

# sample 
# python main.py --dataname shoppers --method tabddpm --mode sample --save_path sample_end_csv/${TASK_NAME}.csv --task_name $TASK_NAME

## dp_sgd
# 如果你走 main.py，可以直接这样用：

# python main.py --dataname shoppers --method tabddpm --mode train_sgd --dp_mode_num 4

# 如果后面要采样对应的 DP-SGD 模型，也可以显式传同一个 mode：

# python main.py --dataname shoppers --method tabddpm --mode sample --dp_mode_num 4 --save_path sample_end_csv/out.csv

####################################################

# tabsyn
# train VAE first
# python main.py --dataname shoppers --method vae --mode train

# after the VAE is trained, train the diffusion model
# python main.py --dataname shoppers --method tabsyn --mode train

# sample
# python main.py --dataname default --method tabsyn --mode sample --save_path sample_end_csv/tabsyn_default_.csv