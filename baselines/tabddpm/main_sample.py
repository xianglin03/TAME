import os
import argparse
from baselines.tabddpm.sample import sample
from baselines.tabddpm.train_sgd import has_explicit_dp_mode_num, resolve_dp_mode_model_save_path, resolve_dp_mode_num

import src


def main(args):
    dataname = args.dataname
    device = f'cuda:0'

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = f'{curr_dir}/configs/{dataname}.toml'
    model_save_path = f'{curr_dir}/ckpt/{dataname}'
    if has_explicit_dp_mode_num(args):
        model_save_path = resolve_dp_mode_model_save_path(
            f'{curr_dir}/ckpt_sgd/{dataname}',
            resolve_dp_mode_num(args),
        )
    real_data_path = f'data/{dataname}'
    sample_save_path = args.save_path

    args.train = True
    
    raw_config = src.load_config(config_path)
    print(args)
    print('----------------------------------')
    print(raw_config)

    ''' 
    Modification of configs
    '''
    print('START SAMPLING')
    
    sample(
        num_samples=raw_config['sample']['num_samples'],
        batch_size=raw_config['sample']['batch_size'],
        disbalance=raw_config['sample'].get('disbalance', None),
        **raw_config['diffusion_params'],
        model_save_path=model_save_path,
        sample_save_path=sample_save_path,
        real_data_path=real_data_path,
        task_type=raw_config['task_type'],
        model_type=raw_config['model_type'],
        model_params=raw_config['model_params'],
        T_dict=raw_config['train']['T'],
        num_numerical_features=raw_config['num_numerical_features'],
        device=device,
        ddim=args.ddim,
        steps=args.steps,
        task_name=args.task_name,
        eval_flag=args.eval_flag 
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type = str, default = 'shoppers')
    parser.add_argument('--gpu', type = int, default=0)
    parser.add_argument('--dp_mode_num', type=int, default=None, choices=[1, 4, 8, 16, 32])
    parser.add_argument('--ddim', action = 'store_true', default = False, help='Whether to use ddim sampling.')
    parser.add_argument('--steps', type=int, default = 1000)

    args = parser.parse_args()
