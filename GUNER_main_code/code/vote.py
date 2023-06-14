# encoding=utf-8

import json
import os,re
import sys
import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append('/home/jw/CCL_Guner2023/GUNER_main_code/code')
from models.GlobalPointer import GlobalPointer1
from utils.data_loader import load_data, EntDataset3
from tools.finetune_args import args
from train import build_transformer_model
import numpy as np
import random

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0")
seed = 1998
same_seeds(seed)
ent2id = json.load(open(args.ent2id_path, encoding="utf-8"))
ent_type_size = len(ent2id)
id2ent = {}
for k, v in ent2id.items():
    id2ent[v] = k

from functools import cmp_to_key
def cmp(x,y):
    if int(x['end_idx'])<int(y['end_idx']):
        return -1
    else:
        return 1

def load_test_data(path):
    datalist = []
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n")
            datalist.append({
                "id": count,
                'text': line,
            })
            count += 1

    D = []
    for d in datalist:
        D.append([d['text']])
    return D

def get_entity_bio(seq, id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0, 1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def merge(ner_loader, model_list):
    print("merge")
    datalist = []
    # candidate_ent = None
    with torch.no_grad():
        for batch in tqdm(ner_loader):
            text, input_ids, attention_mask, segment_ids, mapping = batch
            input_ids, attention_mask, segment_ids = input_ids.to(device), attention_mask.to(
                device), segment_ids.to(device)

            scores_list = []
            for idx, model in enumerate(model_list):
                model.eval()
                scores = model(input_ids, attention_mask, segment_ids)
                scores = scores.data.cpu().numpy()
                scores_list.append(scores)
            # model_all.eval()
            # scores_all = model_all(input_ids, attention_mask, segment_ids)
            # scores_all = scores_all.data.cpu().numpy()
            # 解码
            # print("开始解码-----------------------------------------------------", len(text))
            for i in range(len(text)):
                text_1 = text[i]
                mapping_1 = mapping[i]
                scores = 0
                for score in scores_list:
                    scores += score[i] /args.n_splits

                # scores = scores * 0.5 + scores_all[i] * 0.5
                scores[:, [0, -1]] -= np.inf
                scores[:, :, [0, -1]] -= np.inf
                entities = []
                for l, start, end in zip(*np.where(scores > 0 )):
                    entities.append(
                        {"start_idx": mapping_1[start][0], "end_idx": mapping_1[end][-1], "type": id2ent[l],'score':scores[l][start][end]}
                    )
                entities.sort(key=cmp_to_key(cmp))
                unnested_entities=[]
                for idx,ent in enumerate(entities):
                    if idx==0:
                        candidate_ent=ent
                        continue
                    if candidate_ent['end_idx']>=ent['start_idx']:
                        if ent['score']>candidate_ent['score']:
                            candidate_ent=ent
                    else:
                        unnested_entities.append(candidate_ent)
                        candidate_ent=ent
                unnested_entities.append(candidate_ent)

                datalist.append({
                    'text': text_1,
                    'entities': unnested_entities
                })

    return datalist
        # with open(save_path, 'w', encoding='utf-8') as f:
        #     f.write(json.dumps(datalist, ensure_ascii=False, indent=4))

def save_submit(datalist, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for data in datalist:
            text = data['text']
            entities = data['entities']
            temp_str = ""
            i = 0
            current_idx = 0
            new_entities = []
            if len(entities)>0:
                for sub_entities_index,sub_entities in enumerate(entities):
                    start_idx = sub_entities['start_idx']
                    end_idx = sub_entities['end_idx']
                    if start_idx > current_idx or start_idx == current_idx :
                        current_idx = end_idx
                        new_entities.append(sub_entities)
                    else:
                        continue

                for sub_entities_index,sub_entities in enumerate(new_entities):
                    start_idx = sub_entities['start_idx']
                    end_idx = sub_entities['end_idx']
                    entity_type = sub_entities['type']
                    if i>start_idx:
                        continue
                    for sub_text_indx in range(i,len(text)):
                        if start_idx != sub_text_indx:
                            temp_str += text[sub_text_indx]
                        elif sub_entities_index == len(new_entities)-1:
                            temp_str += "{"+text[start_idx:end_idx+1]+"|"+entity_type+"}"+text[end_idx+1:]
                            i = end_idx+1
                            break          
                        else:
                            temp_str += "{"+text[start_idx:end_idx+1]+"|"+entity_type+"}"
                            i = end_idx+1
                            break

                f.write(temp_str+'\n')
            else:
                f.write(text+'\n')

def end_processing(res_path, end_process_savepath):
    pu_list = ['，' ,'。' ,',','.']
    OFI_list = ['參議中書省事','行尚書省','御史臺臣']
    res_list = []
    with open(res_path,'r') as f:
        with open(end_process_savepath,'w') as f1:
            for lines in f.readlines():
                line = lines.strip("\n")
                for i in OFI_list:
                    if i in line:
                        line = line.replace(i,'{'+i+'|OFI}')
                matches = re.findall(r"\{[^{}]*\}", line)
                temp_list = []
                if matches:
                    for match in matches:
                        for i in pu_list:
                            if i in match.split("|")[0][1:] :
                                print(match)
                                temp_list.append(match)
                    if len(temp_list)>0:
                        temp_str = ""
                        for i in temp_list:
                            ori_match = i.split("|")[0][1:]
                            line = line.replace(i,ori_match)
                            temp_str = line
                        res_list.append(temp_str)
                    else:
                        res_list.append(line)
                else:
                    res_list.append(line)
            for sub_res in res_list:
                if "{之|PER}" in sub_res  :
                    sub_res = sub_res.replace("{之|PER}","之")
                    f1.write(sub_res+"\n")
                elif "{所料|PER}" in sub_res:
                    sub_res = sub_res.replace("{所料|PER}","所料")
                    f1.write(sub_res+"\n")
                elif "{害之慮|PER}" in sub_res:
                    sub_res = sub_res.replace("{害之慮|PER}","害之慮")
                    f1.write(sub_res+"\n")
                elif "節度隴右" in sub_res:
                    sub_res = sub_res.replace("節度隴右","{節度|OFI}隴右")
                    f1.write(sub_res+"\n")
                elif "白正其事" in sub_res :
                    sub_res = sub_res.replace("白正其事","{白|PER}正其事")
                    f1.write(sub_res+"\n")
                else:
                    f1.write(sub_res+"\n")

def predict_kfold_data(datalist, model_dirs):
    model_list = []
    for i in range(len(model_dirs)):
        print(f"----------------------------------------------------{i+1}----------------------------------------------")
        print(model_dirs[i])
        encoder, config, tokenizer = build_transformer_model(args.bert_model_path, model_type='bert')
        model = GlobalPointer1(encoder, ent_type_size, 64).to(device)
        model.load_state_dict(
            torch.load(model_dirs[i], map_location='cuda:0'), False
        )
        model.eval()
        model_list.append(model)
    
    encoder, config, tokenizer = build_transformer_model(args.bert_model_path, model_type='bert')
    # model_all = GlobalPointer1(encoder, ent_type_size, 64).to(device)
    # model_all.load_state_dict(
    #     torch.load('/home/jw/CCL_Guner2023/output/res_model/94.420_swa670_fgm_all_model.pt',
    #                map_location='cuda:0'), False)
    
    ner_train = EntDataset3(
        datalist,
        tokenizer=tokenizer,
        type='test'
    )
    ner_loader_train = DataLoader(ner_train, batch_size=32, collate_fn=ner_train.collate,
                                  shuffle=False, num_workers=10, drop_last=False)
    data_list = merge(ner_loader_train, model_list)
    print("生成完成")
    return data_list


if __name__ == '__main__':

    if args.pseudos:
        os.makedirs(args.save_pseudos_path, exist_ok=True)
        datalist = load_test_data("/home/jw/CCL_Guner2023/output/unlabel_data_res/pseudos.txt")
        print(len(datalist), '----')
        model_dirs = [f'/home/jw/CCL_Guner2023/output/pre_model_200/fgm_all+swa+5fold/model_ancientbert/fold-{i + 1}/checkpoint-saw-100000/model.pt' for i in range(args.n_splits)]
        save_pseudos_path = os.path.join(args.save_pseudos_path,'pseudos+vote_5old.txt')
        data_list = predict_kfold_data(datalist,model_dirs)
        save_submit(data_list, save_pseudos_path)
        end_process_savepath = save_pseudos_path.split(".")[0] + "_end.txt"
        end_processing(save_pseudos_path, end_process_savepath)
    else:
        os.makedirs(args.save_path, exist_ok=True)
        #test_data
        datalist = load_test_data(args.test_file)
        # datalist = load_test_data("/home/jw/CCL_Guner2023/data/test_data/simplified_test_public.txt")
        print(len(datalist), '----')
        model_dirs = [f'/home/jw/CCL_Guner2023/output/pre_model_200/fgm_all+swa+5fold/model_ancientbert/fold-{i + 1}/checkpoint-saw-100000/model.pt' for i in range(args.n_splits)]
        save_path = os.path.join(args.save_path,'fgm_all+swa6-40+5fold.txt')
        data_list = predict_kfold_data(datalist,model_dirs)
        save_submit(data_list, save_path)
        end_process_savepath = save_path.split(".")[0] + "_end.txt"
        end_processing(save_path, end_process_savepath)
