import copy
import os.path
import sys

from torch.utils.data import DataLoader
from transformers import BertModel, BertConfig, BertTokenizer
import os,re
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

def save_json(datalist, save_path):
    DD= []
    with open(save_path, 'w', encoding='utf-8') as f:
        for data in datalist:
            text = data['text']
            new_entities = []
            for sub_entities in  data['entities']:
                temp = [sub_entities['start_idx'],sub_entities['end_idx'],sub_entities['type']]
                new_entities.append(temp)
            DD.append({
                    'text': text,
                    'entity_list': new_entities,
                })
        f.write(json.dumps(DD, ensure_ascii=False, indent=4))

def do_merge(mn=0):
    output = '/home/jw/CCL_Guner2023/output/ancientbert/fgm_all+swa+premodel_alltrain/submissions'
    datas = []
    for file_name in os.listdir(output):
        jsons = load_data(os.path.join(output,file_name))
        datas.append(jsons)
    D= []
    # with open(f'{output}/merge_.json','w') as f:
        # zip(*)将所有输入对象的行转为列
    for rows in map(list,zip(*datas)):
        spoes = []
        dd = rows[0]
        for row in rows: 
            spo_list = row['entity_list']
            for spo in spo_list: 
                spo_s = json.dumps(spo, ensure_ascii=False)
                spoes.append(spo_s)
        if spoes:
            count = Counter(spoes) 
            spo_list = [json.loads(spo_s) for spo_s,c in count.items() if c>=mn] 
            dd['entity_list'] = sorted(spo_list,key = lambda x: x[0])
            
            D.append(dd)
                # f.write(json.dumps(dd,ensure_ascii=False)+"\n")
    return D
        
def load_data(filename):
    """加载数据
    """ 
    data_list = []
    for d in json.load(open(filename, encoding='utf-8')):
        data_list.append(d)
    return data_list

def load_test_data1(path):
    D = []
    for d in json.load(open(path, encoding='utf-8')):
        D.append([d['text']])
    return D

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

def post_processing(train_path,predict_path,save_path):
    '''
    :param file_path:
    :return:
    '''
    num=0
    ans=[]
    trains_aloneB=[]
    #train里的单字符实体
    with open(train_path,'r',encoding='utf-8') as f:
        lines=f.readlines()
        for i in range(len(lines)):
            now_line=lines[i]
            if len(now_line)>3 and now_line[2]=='B':
                if(i==len(lines)-1) or (len(lines[i+1])<3) or lines[i+1][2]!='I' or  lines[i+1][3:]!=now_line[3:]:
                    trains_aloneB.append(now_line)


    with open(predict_path,'r',encoding='utf-8') as f:
        lines=f.readlines()
        for i in range(len(lines)):
            now_line=lines[i]
            if i==0 or i==len(lines)-1:
                continue
            pre_line=lines[i-1]
            nex_line=lines[i+1]
            if now_line[0] in [' ','/','+'] and now_line[2]=='I':
                if len(pre_line)>3 and len(nex_line)>3 and pre_line[2]=='I' and nex_line[2]=='B' and \
                        pre_line[3:]==nex_line[3:] and now_line[3:]==pre_line[3:]:
                    num+=1
                    lines[i]=now_line[0]+" O\n"

        #处理单字符实体
        for i in range(len(lines)):
            now_line=lines[i]
            if len(now_line)>3 and now_line[2]=='B':
                if(i==len(lines)-1) or (len(lines[i+1])<3) or lines[i+1][2]!='I' or  lines[i+1][3:]!=now_line[3:]:
                    if now_line not in trains_aloneB:
                        lines[i]=now_line[0]+' O\n'
                        num+=1
            ans.append(lines[i])

    with open(save_path,'w',encoding='utf-8') as f:
        for line in ans:
            f.write(line)

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
    save_json(datalist,save_path)

def end_processing(res_path, end_process_savepath):
    with open(res_path,'r') as f:
        with open(end_process_savepath,'w') as f1:
            for lines in f.readlines():
                line = lines.strip("\n")
                matches = re.findall(r"\{[^{}]*\}", line)
                temp_list = []
                if matches:
                    for match in matches:
                        if '，' in match or '〈' in match or ']' in match or '。' in match:
                            print(match)
                            temp_list.append(match)
                    if len(temp_list)>0:
                        for i in temp_list:
                            ori_match = i.split("|")[0][1:]
                            line = line.replace(i,ori_match)
                            f1.write(line+"\n")
                    else:
                        f1.write(line+"\n")
                else:
                    f1.write(line+"\n")

# def pred_BIO(path_word:str, path_sample:str, batch_size:int):
def pred_BIO(path_word:str,  batch_size:int):
    model_dirs = ['/home/jw/CCL_Guner2023/output/res_model/94.373_model.pt','/home/jw/CCL_Guner2023/output/res_model/94.420_swa670_fgm_all_model.pt']
    for model_path_index,model_path in enumerate(model_dirs): 
        save_path = os.path.join("/home/jw/CCL_Guner2023/output/ancientbert/fgm_all+swa+premodel_alltrain/submissions",str(model_path_index+1)+'results.json')
        print(model_path)
        print(save_path)
        predict_to_file(path_word, model_path, save_path, batch_size)
    threshold = 1
    datalist = do_merge(threshold)
    # datalist = load_data()
    res_path = "/home/jw/CCL_Guner2023/output/ancientbert/fgm_all+swa+premodel_alltrain/submissions/merge_model_3.txt"
    save_submit(datalist, res_path)
    end_process_savepath = res_path.split(".")[0] + f"_end{str(threshold)}.txt"
    end_processing(res_path, end_process_savepath)


if __name__ == '__main__':
    pred_BIO('/home/jw/CCL_Guner2023/data/test_data/GuNER2023_test_public.txt',1)

