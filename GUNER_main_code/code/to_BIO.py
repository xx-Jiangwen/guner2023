'''
Descripttion: 
version: 
Author: Smallxiaoxin
Date: 2023-04-24 09:58:54
LastEditors: Smallxiaoxin
LastEditTime: 2023-06-05 17:31:59
'''
# coding:utf-8

import json


ori_path ="/home/jw/CCL_Guner2023/output/unlabel_data_res/pseudos+vote_5old_end.txt"
save_BIO_path =  "/home/jw/CCL_Guner2023/output/unlabel_data_res/pseudos+vote_5old+end_BIO.txt"

with open(ori_path,'r') as f:
    with open(save_BIO_path,'w') as f1:
        for lines in f.readlines():
            text = lines.strip("\n")
            words = ""
            for char in text:
                if char == '{':
                    if len(words) > 0:
                        for i in words:
                            f1.write(i+" "+"O"+"\n")
                        words = ""
                elif char == '}':
                    # print(text)
                    entity_name =  words.split("|")[0]
                    entity_type = words.split("|")[1]
                    for entity_name_index,sub_entity_name in enumerate(entity_name):
                        if entity_name_index ==0:
                            f1.write(sub_entity_name+" "+"B-"+entity_type+"\n")
                        else:
                            f1.write(sub_entity_name+" "+"I-"+entity_type+"\n")
                    words = ""
                else:
                    words +=  char
            if len(words) > 0:
                for i in words:
                    f1.write(i+" "+"O"+"\n")
            f1.write("\n")


