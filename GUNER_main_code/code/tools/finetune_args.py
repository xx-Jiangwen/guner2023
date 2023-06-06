'''
Descripttion: 
version: 
Author: Smallxiaoxin
Date: 2023-04-24 09:58:54
LastEditors: Smallxiaoxin
LastEditTime: 2023-06-06 11:26:28
'''
# coding:utf-8

import os
import argparse
def finetune(params=None):

    print("\n Start fine-tuning")

    parser = argparse.ArgumentParser()
    # Required parameters

    parser.add_argument("--model_type", type=str, default='bert')

    parser.add_argument("--train_file", type=str,
                        default='/home/jw/CCL_Guner2023/data/train_json/ori_trian/all_train.json')
    parser.add_argument("--test_file", type=str,
                        default='/home/jw/CCL_Guner2023/data/test_data/GuNER2023_test_public.txt')
    parser.add_argument("--ent2id_path", type=str,
                        default='/home/jw/CCL_Guner2023/data/train_json/ent2id.json')

    parser.add_argument("--bert_model_path", type=str,
                        default='/home/jw/CCL_Guner2023/output/pretrain_model_mlm_200')
    
    parser.add_argument('--output_dir', type=str,
                         default='/home/jw/CCL_Guner2023/output/pre_model_200/fgm_all+swa+5fold/model_ancientbert')
    
    parser.add_argument('--output_dir_best', type=str,
                        default= '/home/jw/CCL_Guner2023/output/pre_model_200/fgm_all+swa+5fold/eval_best')
    
    parser.add_argument('--save_path', type=str,
                        default= '/home/jw/CCL_Guner2023/output/pre_model_200/fgm_all+swa+5fold/submissions')
    
    parser.add_argument('--save_pseudos_path', type=str,
                    default= '/home/jw/CCL_Guner2023/output/unlabel_data_res')

    parser.add_argument('--num_workers', type=int, default=10)

    parser.add_argument("--epoch", type=int, default=45)
    parser.add_argument("--model_eval", type=bool, default=True)
    # 32
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)

    parser.add_argument('--swa_start', type=int, default=6)
    parser.add_argument('--swa_end', type=int, default=40)
    # 4e-5
    parser.add_argument('--lr', type=float, default=4e-5)

    parser.add_argument('--warmup_ratio', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)

    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    parser.add_argument('--do_eval', type=bool, default=True)

    parser.add_argument('--use_fgm', type=bool, default=True)
    parser.add_argument('--use_pgd', type=bool, default=False)
    parser.add_argument('--use_awp', type=bool, default=False)

    parser.add_argument('--adv_k', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--epsilon', type=float, default=0.5)
    parser.add_argument('--awp_lr', type=float, default=0.1)
    parser.add_argument('--awp_eps', type=float, default=0.5)

    parser.add_argument('--model', type=str, default='gp')

    parser.add_argument('--use_ema', type=bool, default=False)
    parser.add_argument('--use_lookahead', type=bool, default=False)
    parser.add_argument('--ema_decay', type=float, default=0.995)

    parser.add_argument('--logging_steps', type=int, default=2250)  # 2250

    parser.add_argument('--seed', type=int, default=1998)  # 1998

    parser.add_argument('--type', type=str, default='testB')
    
    parser.add_argument('--kfold', type=bool, default=True)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--pseudos', type=bool, default=False)
    print(params)
    args = parser.parse_args(args=params)

    return args

args = finetune()

