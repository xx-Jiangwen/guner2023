<!--
 * @Descripttion: 
 * @version: 
 * @Author: Smallxiaoxin
 * @Date: 2023-06-06 11:00:23
 * @LastEditors: Smallxiaoxin
 * @LastEditTime: 2023-06-06 19:15:13
-->
# 预训练阶段
本文采用了动态预训练策略

预训练数据说明：

数据来源如下：https://github.com/mahavivo/core-books 中的“正史”文件夹，详细链接：https://github.com/mahavivo/core-books/tree/main/%E6%AD%A3%E5%8F%B2

具体用到了：《北史》、《后唐书》、《金史》、《梁书》、《辽史》、《明史》、《南史》、《三国志》、《宋书》、《隋书》、《魏书》、《元史》、《史记》

共计137044，并且与测试集做了去重操作，数据位于/data/add_data/unlabel_24_history

预训练脚本：run_pretrain_mlm.py

本次比赛采用BERT+GlobalPointer的网络结构，同时实验了BERT+CRF等多种模型

# 训练预处理

1.数据转换为BIO形式：

python /GUNER_main_code/code/to_BIO.py

2.数据处理为GlobalPointer的输入形式

python /GUNER_main_code/code/data_process.py

# 数据分析阶段

1.分析训练集和测试集的一些基本情况

data_analyse.ipynb

# 模型训练阶段

python /GUNER_main_code/code/train.py

超参数调整：/GUNER_main_code/code/tools/finetune_args.py

# 模型预测阶段：

不加交叉验证：

python GUNER_main_code/code/submit_res_1.py

交叉验证：

python GUNER_main_code/code/vote.py

# 伪标签策略

本次比赛采用了伪标签手段

原始无标签数据：/home/jw/CCL_Guner2023/data/unlabel_data_res/pseudos.txt

匹配训练数据样式：/home/jw/CCL_Guner2023/data/unlabel_data_res/all_pseudos.json



