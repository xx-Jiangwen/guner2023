'''
Descripttion: 
version: 
Author: Smallxiaoxin
Date: 2023-05-17 21:01:46
LastEditors: Smallxiaoxin
LastEditTime: 2023-05-29 15:57:48
'''
import torch
import json,os
import copy,random
from transformers import BertModel, BertTokenizer,  BertConfig
from utils.functions_utils import swa
from tools.finetune_args import args
from tools.common import init_logger, logger
from models.loss import loss_fun
from models.GlobalPointer import GlobalPointer1, MetricsCalculator
from tqdm import tqdm
from utils.data_loader import load_data, EntDataset3
from torch.utils.data import DataLoader
import time
device = torch.device("cuda:0")


def build_transformer_model(bert_model_path, model_type='bert'):
    tokenizer = BertTokenizer.from_pretrained(bert_model_path, do_lower_case=True)
    encoder, config = None, None
    if model_type == 'bert':
        config = BertConfig.from_pretrained(bert_model_path, num_labels=ent_type_size)
        config.output_hidden_states = True
        encoder = BertModel.from_pretrained(bert_model_path, config=config)
    return encoder, config, tokenizer


ent2id = json.load(open(args.ent2id_path, encoding="utf-8"))
ent_type_size = len(ent2id)
encoder, config, tokenizer = build_transformer_model(args.bert_model_path, args.model_type)
datalist = load_data(args.train_file)   #加载数据 [[text,(start, end, ent2id[label]),(start, end, ent2id[label]),(start, end, ent2id[label])]]
# random.shuffle(datalist)

logger.info(f"数量{len(datalist)}")
logger.info(f"训练数量:{int(len(datalist)*0.8)}")

#训练集:验证集 = 8:2
ner_train = EntDataset3(datalist[:int(len(datalist)*0.8)], tokenizer=tokenizer)
ner_loader_train = DataLoader(ner_train, batch_size=args.batch_size, num_workers=args.num_workers,
                                collate_fn=ner_train.collate, shuffle=True, pin_memory=True)

ner_eval = EntDataset3(datalist[-int(len(datalist)*0.2):], tokenizer=tokenizer,type='eval')
ner_loader_eval = DataLoader(ner_eval, batch_size=args.batch_size * 8, num_workers=args.num_workers,
                                collate_fn=ner_eval.collate, shuffle=False, pin_memory=True)

def evaluate(model, ner_loader_evl, metrics):
    model.eval()

    eval_metric = {}
    total_loss = 0  # 验证集中所有数据的loss之和
    total_f1_, total_precision_, total_recall_ = 0., 0., 0.
    for batch in tqdm(ner_loader_evl, desc="Evaluation"):
        raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
        input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(device), \
                                                         segment_ids.to(device), labels.to(device)

        with torch.no_grad():  #关闭梯度计算，以减少内存消耗并提高推理速度
            logits = model(input_ids, attention_mask, segment_ids)
        eval_loss = loss_fun(logits, labels)
        f1, p, r = metrics.get_evaluate_fpr(logits, labels)
        total_f1_ += f1
        total_precision_ += p
        total_recall_ += r
        total_loss += eval_loss
    avg_eval_loss = total_loss / (len(ner_loader_evl))
    avg_f1 = total_f1_ / (len(ner_loader_evl))
    avg_precision = total_precision_ / (len(ner_loader_evl))
    avg_recall = total_recall_ / (len(ner_loader_evl))

    eval_metric['f1'], eval_metric['precision'], eval_metric['recall'], eval_metric['eval_loss']= avg_f1, avg_precision, avg_recall,avg_eval_loss

    return eval_metric


encoder, config, tokenizer = build_transformer_model(args.bert_model_path, args.model_type)
logger.info("GlobalPointer")
model = GlobalPointer1(encoder, ent_type_size, 64).to(device)

swa_raw_model = copy.deepcopy(model)   #swa

if __name__ == '__main__':
    swa_start = 2
    swa_end = 40
    if args.kfold:
        model_dirs = [f'/home/jw/CCL_Guner2023/output/pre_model_200/fgm_all+swa+5fold/model_ancientbert/fold-{i + 1}' for i in range(args.n_splits)]
        for choose_swa_path in model_dirs:
            swa(swa_raw_model, choose_swa_path, swa_start=swa_start,swa_end= swa_end)
            metrics = MetricsCalculator()
            model.load_state_dict(torch.load(os.path.join(choose_swa_path, 'checkpoint-saw-100000/model.pt'),
                                                map_location='cuda:0'))

            metric = evaluate(model, ner_loader_eval, metrics)
            f1_score = metric['f1']
            recall = metric['recall']
            precision = metric['precision']
            time_ = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
            init_logger(log_file=choose_swa_path + f'/{args.model_type}-{time_}-{swa_start}-{swa_end}.txt')
            logger.info("\n >> Average: \t Eval f1 : {}\t Precision : {}\t Recall : {}\t ".
                        format(f1_score, recall, precision))
    else:
        choose_swa_path = '/home/jw/CCL_Guner2023/output/pre_model_50/fgm_all+swa/model_ancientbert'
        swa(swa_raw_model, choose_swa_path, swa_start=swa_start,swa_end= swa_end)
        metrics = MetricsCalculator()
        model.load_state_dict(torch.load(os.path.join(choose_swa_path, 'checkpoint-saw-100000/model.pt'),
                                            map_location='cuda:0'))

        metric = evaluate(model, ner_loader_eval, metrics)

        f1_score = metric['f1']
        recall = metric['recall']
        precision = metric['precision']
        time_ = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        init_logger(log_file=choose_swa_path + f'/{args.model_type}-{time_}-{swa_start}-{swa_end}.txt')
        logger.info("\n >> Average: \t Eval f1 : {}\t Precision : {}\t Recall : {}\t ".
                    format(f1_score, recall, precision))
