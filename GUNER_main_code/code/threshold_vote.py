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

def vote(entities_list, threshold=None):
    """
    实体级别的投票方式  (entity_type, entity_start, entity_end, entity_text)
    :param entities_list: 所有模型预测出的一个文件的实体
    :param threshold:大于70%模型预测出来的实体才能被选中
    :return:[{type:[(start, end), (start, end)]}, {type:[(start, end), (start, end)]}]
    """
    threshold_nums = int(len(entities_list) * threshold)
    entities_dict = defaultdict(int)
    entities = defaultdict(list)

    for _entities in entities_list:
        for _type in _entities:
            for _ent in _entities[_type]:
                entities_dict[(_type, _ent[0], _ent[1])] += 1

    for key in entities_dict:
        print(entities_dict[key])
        if entities_dict[key] >= threshold_nums:
            entities[key[0]].append((key[1], key[2]))

    return entities

def predict(ner_loader, model_list):
    datalist = []
    with torch.no_grad():
        for batch in tqdm(ner_loader):
            text, input_ids, attention_mask, segment_ids, mapping = batch
            input_ids, attention_mask, segment_ids = input_ids.to(device), attention_mask.to(
                device), segment_ids.to(device)

            scores_list = []
            for idx, model in enumerate(model_list):
                model.eval()

                scores = model(input_ids, attention_mask, segment_ids).data.cpu().numpy()
                scores_list.append(scores)

            # 解码
            # print("开始解码", len(text))
            for i in range(len(text)):
                text_1 = text[i]
                mapping_1 = mapping[i]
                entities_ls = []
                for scores in scores_list:
                    scores = scores[i]
                    scores[:, [0, -1]] -= np.inf
                    scores[:, :, [0, -1]] -= np.inf
                    # print(scores.shape)
                    predict_entities = {}
                    for l, start, end in zip(*np.where(scores > 0)):
                        if id2ent[l] not in predict_entities:
                            predict_entities[id2ent[l]] = [(mapping_1[start][0], mapping_1[end][-1])]
                        else:
                            predict_entities[id2ent[l]].append((mapping_1[start][0], mapping_1[end][-1]))

                    entities_ls.append(predict_entities)

                entities = vote(entities_ls, 0.01)
                tmp = []
                for key in entities:
                    for ent in entities[key]:
                        tmp.append(
                            [ent[0], ent[-1], key]
                        )
                datalist.append({
                    'text': text_1,
                    'entity_list': tmp
                })

    return datalist
    # with open(save_path, 'w', encoding='utf-8') as f:
    #     f.write(json.dumps(datalist, ensure_ascii=False, indent=4))

def end_processing(res_path, end_process_savepath):
    pu_list = ['，' ,'。' ,',','.']
    with open(res_path,'r') as f:
        with open(end_process_savepath,'w') as f1:
            for lines in f.readlines():
                line = lines.strip("\n")
                matches = re.findall(r"\{[^{}]*\}", line)
                temp_list = []
                if matches:
                    for match in matches:
                        for char in match:
                            if char in pu_list:
                                print(match)
                                temp_list.append(match)
                    if len(temp_list)>0:
                        temp_str = ""
                        for i in temp_list:
                            ori_match = i.split("|")[0][1:]
                            line = line.replace(i,ori_match)
                            temp_str = line
                        f1.write(temp_str+"\n")
                    else:
                        f1.write(line+"\n")
                else:
                    f1.write(line+"\n")
def save_submit(datalist, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for data in datalist:
            text = data['text']
            entities = data['entity_list']
            temp_str = ""
            i = 0
            current_idx = 0
            new_entities = []
            if len(entities)>0:
                for sub_entities_index,sub_entities in enumerate(entities):
                    start_idx = sub_entities[0]
                    end_idx = sub_entities[1]
                    if start_idx > current_idx or start_idx == current_idx :
                        current_idx = end_idx
                        new_entities.append(sub_entities)
                    else:
                        continue

                for sub_entities_index,sub_entities in enumerate(new_entities):
                    start_idx = sub_entities[0]
                    end_idx = sub_entities[1]
                    entity_type = sub_entities[2]
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

def predict_data(datalist):
    model_list = []

    # 遍历目录及其子目录下的所有文件和文件夹
    for root, dirs, files in os.walk("/home/jw/CCL_Guner2023/output/res_model"):
        # 处理当前目录下的文件
        for file in files:
            file_path = os.path.join(root, file)
            encoder, config, tokenizer = build_transformer_model(args.bert_model_path, model_type='bert')
            model = GlobalPointer1(encoder, ent_type_size, 64).to(device)
            model.load_state_dict(
                torch.load(file_path, map_location='cuda:0')
            )
        model_list.append(model)

    ner_train = EntDataset3(
        datalist,
        tokenizer=tokenizer,
        type='test'
    )
    ner_loader_train = DataLoader(ner_train, batch_size=1, collate_fn=ner_train.collate,
                                  shuffle=False, num_workers=10, drop_last=False)

    data_list = predict(ner_loader_train, model_list)
    return data_list


if __name__ == '__main__':
    os.makedirs(args.save_path, exist_ok=True)
    datalist = load_test_data(args.test_file)
    print(len(datalist), '----')
    save_path = os.path.join(args.save_path,'vote.txt')
    data_list = predict_data(datalist)
    save_submit(data_list, save_path)
    end_process_savepath = save_path.split(".")[0] + "_end.txt"
    end_processing(save_path, end_process_savepath)