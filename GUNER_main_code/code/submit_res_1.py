import copy
import os.path
import sys,re

from torch.utils.data import DataLoader
from transformers import BertModel, BertConfig, BertTokenizer
import os
from collections import Counter
base_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(base_dir, 'code/models'))
from models.GlobalPointer import GlobalPointer1
import json
import torch
import numpy as np
from tqdm import tqdm
from tools.finetune_args import args
# sys.path.append(os.path.join(base_dir, 'code/tools'))
# from finetune_args import args
sys.path.append(os.path.join(base_dir, 'code/utils'))
from utils.test_data_loader import EntDataset3

# base_dir = os.path.dirname(__file__)
# sys.path.append(os.path.join(base_dir, './'))
# sys.path.append(os.path.join(base_dir, 'code/pre_model'))
# from code.configuration_nezha import NeZhaConfig
# from code.modeling_nezha import NeZhaModel

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


same_seeds(1998)
# bert_model_path = '/home/jw/CCL_Guner2023/data/pretrain_model'
bert_model_path = args.bert_model_path
model_type = 'bert'
model_ttt = 'gp'

# ent2id_path = '/home/jw/CCL_Guner2023/data/train_json/ent2id.json'
ent2id_path = args.ent2id_path
device = torch.device("cuda:0")

ent2id = json.load(open(ent2id_path, encoding="utf-8"))
ent_type_size = len(ent2id)
id2ent = {}
for k, v in ent2id.items():
    id2ent[v] = k

def build_transformer_model(bert_model_path, model_type='bert'):
    tokenizer = BertTokenizer.from_pretrained(bert_model_path, do_lower_case=True)
    if model_type == 'bert':
        config = BertConfig.from_pretrained(bert_model_path, num_labels=ent_type_size)
        config.output_hidden_states = True
        encoder = BertModel.from_pretrained(bert_model_path, config=config)

    return encoder, config, tokenizer

from functools import cmp_to_key

def cmp(x,y):
    if int(x['end_idx'])<int(y['end_idx']):
        return -1
    else:
        return 1

def predict(ner_loader, tokenizer, model):
    # 新加
    # low_frequency_type = ['51', '33', '42', '24', '53', '35', '26']
    # 结束
    datalist = []
    id=0
    candidate_ent = None
    for batch in tqdm(ner_loader):
        text, input_ids, attention_mask, segment_ids, mapping = batch
        input_ids, attention_mask, segment_ids = input_ids.to(device), attention_mask.to(
            device), segment_ids.to(device)

        scores = model(input_ids, attention_mask, segment_ids)
        scores = scores[0].data.cpu().numpy()   #scores[0]就是取出这个Tensor在batch中的第一个元素，所以scores[0]的shape就是(sequence_length, num_labels)，即第一个样本的所有标签的得分值。
        
        # candidate_ent = []
        text = text[0]
        mapping = mapping[0]
        scores[:, [0, -1]] -= np.inf
        scores[:, :, [0, -1]] -= np.inf
        entities = []
        for l, start, end in zip(*np.where(scores > 0 )):
            entities.append(
                {"start_idx": mapping[start][0], "end_idx": mapping[end][-1], "type": id2ent[l],'score':scores[l][start][end]}
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
            'text': text,
            'entities': unnested_entities
        })
        id+=1
    return datalist

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

def end_processing(res_path, end_process_savepath):
    pu_list = ['，' ,'。' ,',','.','所料','之']
    OFI_list = ['參議中書省事','參知政事','行尚書省','御史臺臣']
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
                        if match.split("|")[0][1:] in pu_list:
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

def predict_to_file(file_path, model_path, save_path,batch_size):
    encoder, config, tokenizer = build_transformer_model(bert_model_path, model_type=model_type)
    model = GlobalPointer1(encoder, ent_type_size, 64).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location='cuda:0'), False
    )
    model.eval()

    ner_train = EntDataset3(
        load_test_data(file_path),
        tokenizer=tokenizer,
        type='test'
    )
    ner_loader_train = DataLoader(ner_train, batch_size=batch_size, collate_fn=ner_train.collate,
                                  shuffle=False, num_workers=10, drop_last=False)
    datalist = predict(ner_loader_train, tokenizer, model)
    save_submit(datalist, save_path)


# def pred_BIO(path_word:str, path_sample:str, batch_size:int):
def pred_BIO(path_word:str,  batch_size:int):
    model_path = '/home/jw/CCL_Guner2023/output/pre_model_200/fgm_all+swa+pseudos1000/model_ancientbert/checkpoint-saw-100000/model.pt'
    # model_path = '/home/jw/CCL_Guner2023/output/res_model/94.420_swa670_fgm_all_model.pt'
    print(model_path)
    save_path = "/home/jw/CCL_Guner2023/output/pre_model_200/fgm_all+swa+pseudos1000/submissions/fgm_all+swa+pre_model_200+pseudos1000.txt"
    # save_path = "/home/jw/CCL_Guner2023/output/unlabel_data_res/pseudos_data.txt"
    predict_to_file(path_word, model_path, save_path, batch_size)
    end_process_savepath = save_path.split(".")[0] + "_end.txt"
    end_processing(save_path, end_process_savepath)


if __name__ == '__main__':
    # pred_BIO('/home/jw/CCL_Guner2023/output/unlabel_data_res/pseudos.txt',1)
    pred_BIO("/home/jw/CCL_Guner2023/data/test_data/GuNER2023_test_public.txt",1)

