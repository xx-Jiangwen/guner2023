# coding:utf-8
# !/usr/bin/python

import gc
import os
import copy
import shutil
import json
import time
import torch
import random
import numpy as np
from torch import nn
import torch.nn.functional as F
from prettytable import PrettyTable
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertModel, RobertaModel, RobertaConfig, BertConfig, BertTokenizer,AutoTokenizer, AutoModel,AutoConfig

from models.loss import loss_fun
from tools.finetune_args import args
from utils.functions_utils import swa
from tools.common import init_logger, logger
from utils.tools import AWP
import sys
sys.path.append('/home/jw/CCL_Guner2023/GUNER_main_code/code')
from callback.adversarial import FGM, PGD, EMA
from callback.optimizater.lookahead import Lookahead
from pre_model.modeling_nezha import NeZhaModel
from pre_model.configuration_nezha import NeZhaConfig
from utils.data_loader import load_data, EntDataset3
from models.GlobalPointer import GlobalPointer1, MetricsCalculator

device = torch.device("cuda:0")

ent2id = json.load(open(args.ent2id_path, encoding="utf-8"))
ent_type_size = len(ent2id)


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_transformer_model(bert_model_path, model_type='bert'):
    tokenizer = BertTokenizer.from_pretrained(bert_model_path, do_lower_case=True)
    encoder, config = None, None
    if model_type == 'bert':
        config = BertConfig.from_pretrained(bert_model_path, num_labels=ent_type_size)
        config.output_hidden_states=True
        encoder = BertModel.from_pretrained(bert_model_path, config=config)
    if model_type == 'sikubert':
        config = NeZhaConfig.from_pretrained(bert_model_path)
        encoder = NeZhaModel.from_pretrained(bert_model_path, config=config)
    if model_type == 'roberta':
        config = RobertaConfig.from_pretrained(bert_model_path, num_labels=ent_type_size)
        encoder = RobertaModel.from_pretrained(bert_model_path, config=config)
    return encoder, config, tokenizer


def build_optimizer_and_scheduler(model, t_total, T_mult=1, rewarm_epoch_num=1):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon, weight_decay=args.weight_decay)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    # optimizer = Lookahead(optimizer, 5, 0.5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, t_total // args.epoch * rewarm_epoch_num,
                                            T_mult, eta_min=5e-6, last_epoch=-1)
    return optimizer, scheduler


def save_model(model, global_step):
    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))   #
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    model_to_save = (model.module if hasattr(model, "module") else model)
    print(f'Saving model & optimizer & scheduler checkpoint to {output_dir}')
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))


def save_model_best(model):
    output_dir = os.path.join(args.output_dir_best)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    model_to_save = (model.module if hasattr(model, "module") else model)
    print(f'Saving model & optimizer & scheduler checkpoint to {output_dir}')
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))

def save_model_best_kfold(model,fold):
    output_dir = os.path.join(args.output_dir_best)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    model_to_save = (model.module if hasattr(model, "module") else model)
    print(f'Saving model & optimizer & scheduler checkpoint to {output_dir}')
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, str(fold)+'model.pt'))


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


def kl_ner(logist1, logist2):
    return torch.sum((F.sigmoid(logist1) - F.sigmoid(logist2)) * (logist1 - logist2), dim=[1, 2, 3])



def train_kfold_step():

    # train_data and val_data
    datalist = load_data(args.train_file)
    if args.pseudos:  #是否使用伪标签
        unlabeled_path = '/home/jw/CCL_Guner2023/output/unlabel_data_res/all_pseudos.json'
        unlabeled = load_data(unlabeled_path)

        logger.info(f"未标注伪标签数量{len(unlabeled)}, path: {unlabeled_path}")
        
        datalist = unlabeled + datalist
    kfold = KFold(n_splits=args.n_splits, random_state=args.seed, shuffle=True)

    # k-fold
    logger.info(f"开始{args.n_splits}折训练")
    output_dir = args.output_dir
    for fold, (train_index, val_index) in enumerate(kfold.split(range(len(datalist)))):
        logger.info(f"第{fold + 1}折")
        args.output_dir = os.path.join(output_dir, f'fold-{fold + 1}')
        os.makedirs(args.output_dir, exist_ok=True)

        # tokenizer
        encoder, config, tokenizer = build_transformer_model(args.bert_model_path, args.model_type)

        train_data = np.array(datalist)[train_index]
        val_data = np.array(datalist)[val_index]
        logger.info(f"训练集个数:{len(train_data)}, 验证集个数:{len(val_data)}")
        ner_train = EntDataset3(train_data, tokenizer=tokenizer)
        ner_loader_train = DataLoader(ner_train, batch_size=args.batch_size, collate_fn=ner_train.collate,
                                      shuffle=True, num_workers=args.num_workers)

        ner_evl = EntDataset3(val_data, tokenizer=tokenizer,type='eval')
        ner_loader_evl = DataLoader(ner_evl, batch_size=args.batch_size * 8, collate_fn=ner_evl.collate,
                                    shuffle=False, num_workers=0)

        # GP MODEL
        logger.info("GlobalPointer")
        model = GlobalPointer1(encoder, ent_type_size, 64).to(device)

        swa_raw_model = copy.deepcopy(model)

        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        t_total = len(ner_loader_train) * args.epoch
        optimizer, scheduler = build_optimizer_and_scheduler(model, t_total)
        logger.info("Training/evaluation parameters %s", args)

        if args.use_ema:
            ema = EMA(model.module if hasattr(model, 'module') else model, decay=args.ema_decay)
            ema.register()

        save_steps = t_total // args.epoch
        args.logging_steps = save_steps

        metrics = MetricsCalculator()
        global_steps, total_loss, cur_avg_loss =  0, 0., 0.
        best_f1,pre,rec = 0., 0., 0.


        model.train()

        for epoch in range(args.epoch):

            train_iterator = tqdm(ner_loader_train, desc=f'Fold: {fold + 1} Epoch : {epoch + 1}', total=len(ner_loader_train))

            for batch in tqdm(ner_loader_train):

                model.zero_grad()

                raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
                input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(
                    device), segment_ids.to(device), labels.to(device)
                logits = model(input_ids, attention_mask, segment_ids)
                loss = loss_fun(logits, labels)
    
                loss.backward()

                # if args.use_fgm and epoch % 2 == 0:
                if args.use_fgm:
                    fgm = FGM(model, emb_name="word_embeddings", epsilon=args.epsilon)
                    fgm.attack()
                    logits = model(input_ids, attention_mask, segment_ids)
                    loss_adv = loss_fun(logits, labels)
                    loss_adv.backward()
                    fgm.restore()

                if args.use_pgd and epoch % 2 != 0:
                    pgd = PGD(model, emb_name="word_embeddings", epsilon=args.epsilon, alpha=args.alpha)
                    pgd.backup_grad()
                    for _t in range(args.adv_k):
                        pgd.attack(is_first_attack=(_t == 0))
                        if _t != args.adv_k - 1:
                            model.zero_grad()
                        else:
                            pgd.restore_grad()
                        logits = model(input_ids, attention_mask, segment_ids)
                        loss_adv = loss_fun(logits, labels)
                        loss_adv.backward()
                    pgd.restore()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()

                if args.use_ema:
                    ema.update()

                scheduler.step()
                optimizer.zero_grad()
                train_iterator.set_postfix_str(f'running training loss: {loss.item():.4f}')
                global_steps += 1

                if global_steps % args.logging_steps == 0:

                    if args.do_eval:

                        logger.info("\n >> Start evaluating ... ... ")

                        if args.use_ema:
                            ema.apply_shadow()

                        metric = evaluate(model, ner_loader_evl, metrics)

                        f1_score = metric['f1']
                        recall = metric['recall']
                        precision = metric['precision']
                        eval_loss = metric['eval_loss']

                        if f1_score > best_f1:
                           best_f1 = f1_score
                           save_model_best_kfold(model,fold+1)

                        logger.info("Epoch : {}\t Eval f1 : {}\t Precision : {}\t Recall : {}\t Eval loss : {} ".
                                    format(epoch + 1, f1_score, precision, recall,eval_loss))

                        if args.use_ema:
                            ema.restore()

                        model.train()

                if global_steps % save_steps == 0:
                    save_model(model, global_steps)
        logger.info("Best  Eval f1 : {} ".format(best_f1))
    
        swa(swa_raw_model, args.output_dir, swa_start=args.swa_start)

        torch.cuda.empty_cache()
        gc.collect()

        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'checkpoint-saw-100000/model.pt'),
                                         map_location='cuda:0'))

        metric = evaluate(model, ner_loader_evl, metrics)

        f1_score = metric['f1']
        recall = metric['recall']
        precision = metric['precision']
        # file_dirs = os.listdir(args.output_dir)
        # for d in file_dirs:
        #     if 'saw' in d or 'txt' in d:
        #         continue
        #     else:
        #         print(f"remove {d} from {args.output_dir}")
        #         shutil.rmtree(os.path.join(args.output_dir, d))
        logger.info("\n >> Average: \t Eval f1 : {}\t Precision : {}\t Recall : {}\t ".
                    format(f1_score, recall, precision))

        optimizer.zero_grad()
        del model, optimizer, scheduler, swa_raw_model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"fold-{fold + 1}train done")
    logger.info("train done")



def train_step():
    encoder, config, tokenizer = build_transformer_model(args.bert_model_path, args.model_type)

    datalist = load_data(args.train_file)   #加载数据 [[text,(start, end, ent2id[label]),(start, end, ent2id[label]),(start, end, ent2id[label])]]
    # random.shuffle(datalist)
    if args.pseudos:  #是否使用伪标签
        unlabeled_path = '/home/jw/CCL_Guner2023/output/unlabel_data_res/all_train+pseudos_1000.json'
        unlabeled = load_data(unlabeled_path)

        logger.info(f"未标注伪标签数量{len(unlabeled)}, path: {unlabeled_path}")
        
        datalist = unlabeled + datalist

    logger.info(f"数量{len(datalist)}")
    logger.info(f"训练数量:{int(len(datalist)*0.8)}")
    
    #训练集:验证集 = 8:2
    if args.model_eval:
        ner_train = EntDataset3(datalist[:int(len(datalist)*0.8)], tokenizer=tokenizer)
        ner_loader_train = DataLoader(ner_train, batch_size=args.batch_size, num_workers=args.num_workers,
                                    collate_fn=ner_train.collate, shuffle=True, pin_memory=True)

        ner_eval = EntDataset3(datalist[-int(len(datalist)*0.2):], tokenizer=tokenizer,type='eval')
        ner_loader_eval = DataLoader(ner_eval, batch_size=args.batch_size * 8, num_workers=args.num_workers,
                                    collate_fn=ner_eval.collate, shuffle=False, pin_memory=True)
    else:
        ner_train = EntDataset3(datalist, tokenizer=tokenizer)
        ner_loader_train = DataLoader(ner_train, batch_size=args.batch_size, num_workers=args.num_workers,
                                    collate_fn=ner_train.collate, shuffle=True, pin_memory=True)


    # default set to gp

    logger.info("GlobalPointer")
    model = GlobalPointer1(encoder, ent_type_size, 64).to(device)

    swa_raw_model = copy.deepcopy(model)   #swa

    t_total = len(ner_loader_train) * args.epoch
    optimizer, scheduler = build_optimizer_and_scheduler(model, t_total)
    logger.info("Training/evaluation parameters %s", args)

    if args.use_ema:
        ema = EMA(model.module if hasattr(model, 'module') else model, decay=args.ema_decay)
        ema.register()

    save_steps = t_total // args.epoch
    args.logging_steps = save_steps
    metrics = MetricsCalculator()
    global_steps, total_loss, cur_avg_loss, best_f1 = 0, 0., 0., 0.

    model.train()

    if args.use_awp:
    # 初始化AWP
        awp = AWP(model, loss_fun, optimizer, adv_lr=args.awp_lr, adv_eps=args.awp_eps)

    for epoch in range(args.epoch):

        train_iterator = tqdm(ner_loader_train, desc=f'Epoch : {epoch + 1}', total=len(ner_loader_train))
        for batch in train_iterator:

            model.zero_grad()
            raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
            input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(
                device), segment_ids.to(device), labels.to(device)
            logits = model(input_ids, attention_mask, segment_ids)
            loss = loss_fun(logits, labels)   #multilabel_categorical_crossentropy 多类别不均衡的损失
            loss.backward()

            # if args.use_fgm and epoch % 2 == 0:
            if args.use_fgm:
                fgm = FGM(model, emb_name="word_embeddings", epsilon=args.epsilon)
                fgm.attack()
                logits = model(input_ids, attention_mask, segment_ids)
                loss_adv = loss_fun(logits, labels)
                loss_adv.backward()
                fgm.restore()

            if args.use_awp:
                loss = awp.attack_backward(input_ids, attention_mask, segment_ids, labels)
                loss.backward()
                awp._restore() 
            # if args.use_pgd and epoch % 2 != 0:
            if args.use_pgd:
                pgd = PGD(model, emb_name="word_embeddings", epsilon=args.epsilon, alpha=args.alpha)
                pgd.backup_grad()
                for _t in range(args.adv_k):
                    pgd.attack(is_first_attack=(_t == 0))
                    if _t != args.adv_k - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    logits = model(input_ids, attention_mask, segment_ids)
                    loss_adv = loss_fun(logits, labels)
                    loss_adv.backward()
                pgd.restore()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)   #梯度裁剪

            optimizer.step()

            if args.use_ema:
                ema.update()

            scheduler.step()
            optimizer.zero_grad()

            train_iterator.set_postfix_str(f'running training loss: {loss.item():.4f}')
            global_steps += 1

            if global_steps % args.logging_steps == 0:

                if args.model_eval:

                    logger.info("\n >> Start evaluating ... ... ")

                    if args.use_ema:
                        ema.apply_shadow()
                    metric = evaluate(model, ner_loader_eval, metrics)

                    f1_score = metric['f1']
                    recall = metric['recall']
                    precision = metric['precision']
                    eval_loss = metric['eval_loss']

                    if f1_score > best_f1:
                        best_f1 = f1_score
                        save_model_best(model)

                    logger.info("Epoch : {}\t Eval f1 : {}\t Precision : {}\t Recall : {}\t Eval loss : {} ".
                                format(epoch + 1, f1_score, precision, recall,eval_loss))

                    if args.use_ema:
                        ema.restore()

                    model.train()

            if global_steps % save_steps == 0:
                save_model(model, global_steps)
    logger.info("Best  Eval f1 : {} ".format(best_f1))
    
    swa(swa_raw_model, args.output_dir, swa_start=args.swa_start,swa_end=args.swa_end)

    torch.cuda.empty_cache()
    gc.collect()

    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'checkpoint-saw-100000/model.pt'),
                                     map_location='cuda:0'))

    if args.model_eval:
        metric = evaluate(model, ner_loader_eval, metrics)

        f1_score = metric['f1']
        recall = metric['recall']
        precision = metric['precision']

        logger.info("\n >> Average: \t Eval f1 : {}\t Precision : {}\t Recall : {}\t ".
                    format(f1_score, recall, precision))
    # file_dirs = os.listdir(args.output_dir)
    # for d in file_dirs:
    #     if 'saw' in d or 'txt' in d:
    #         continue
    #     else:
    #         print(f"remove {d} from {args.output_dir}")
    #         shutil.rmtree(os.path.join(args.output_dir, d))
    optimizer.zero_grad()

    del model, optimizer, scheduler, swa_raw_model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("train done")


def main():
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir_best, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)

    time_ = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    init_logger(log_file=args.output_dir + f'/{args.model_type}-{time_}.txt')

    logger.info("\n >>> Arguments ")
    show_table = PrettyTable(['epoch', 'max_length', 'batch_size',
                              'fgm', 'pgd', 'ema', 'lookahead', 'swa_start','swa_end',
                              'warmup_ratio', 'weight_decay'])

    show_table.add_row([args.epoch, args.max_length, args.batch_size,
                        args.use_fgm, args.use_pgd, args.use_ema, args.use_lookahead, args.swa_start,args.swa_end,
                        args.warmup_ratio, args.weight_decay])
    logger.info(show_table)

    same_seeds(args.seed)
    # train_step()
    if args.kfold:
        train_kfold_step()
    else:
        train_step()


if __name__ == '__main__':
    print(args)
    main()
