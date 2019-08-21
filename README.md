
LensNLP 深度学习
---

### 预置预训练模型


| 模型 | 语言 | 数据集 | F1-score |
| -------------------------------  | ---  | ----------- | ---------------- |
| 命名实体识别 高精度 |中文 |  Elensdata  | 93.2% |
| 命名实体识别 高性能 |中文 |   Elensdata | 90.1%   |
| 命名实体识别 高精度 |英文 |  ConLL03  | 93.07% |
| 命名实体识别 高性能 |英文|  ConLL03| 90.4%|
| 命名实体识别 高精度 |维吾尔语  | Elensdata  | 92.8% |
| 文本分类 | 中文  | 13类新闻 | 78% |
|文本分类 |维吾尔语  |  8类新闻  | 86.1% |
|情感分析 |维吾尔语  |   Elensdata | 86.42% |
|情感分析 |英文  |  Elensdata  | 77.86% |
|情感分析 |中文 | Elensdata |75.8%|

预训练模型下载地址为 http://36.112.85.6:4501/source/ 申请新账号请联系Hongjin。
## 快速开始


### 安装


```console
pip install lensnlp-0.0.0-py3-none-any.whl
```


### 使用示例

#### 利用预训练模型进行预测
目前实体识别支持中文(cn)，英文(en)，维吾尔语(uy)的通用实体识别，更改ner()中的语言类别即可。
```python
from lensnlp import ner

cn_tagger = ner('cn')
en_tagger = ner('en')

cn_text = '黄圆圆喜欢周阳'
en_text = 'Yuanyuan Huang is fond of Yang Zhou'

print(cn_tagger.predict(cn_text))
print(en_tagger.predict(en_text))
```

结果为: 

```console
[{'text': '黄圆圆喜欢周阳', 'labels': [], 'entities': [{'text': '黄圆圆', 
'start_pos': 0, 'end_pos': 3, 'type': 'PER', 'confidence': 0.9955726663271586}, 
{'text': '周阳', 'start_pos': 5, 'end_pos': 7, 
'type': 'PER', 'confidence': 0.8638311624526978}]}]

[{'text': 'Yuanyuan Huang is fond of Yang Zhou', 'labels': [], 'entities': 
[{'text': 'Yuanyuan Huang', 'start_pos': 0, 'end_pos': 14, 'type': 'PER', 
'confidence': 0.9937465190887451}, {'text': 'Yang Zhou', 'start_pos': 26,
 'end_pos': 35, 'type': 'PER', 'confidence': 0.9306863844394684}]}]
```
目前文本分类支持多种模型，在clf()中更改语言/维吾尔语：uy，英语：en，中文：cn/，和任务类型/分类：clf，情感识别：emo/即可调用对应模型：
```python
from lensnlp import clf
cn_clf = clf('cn_clf')
# 输入可以为str或者list(str)
text = '中国科学院国家天文台研究员陆由俊对科技日报记者介绍说：“所有超大质量黑洞都能吞噬附近物质，\
        吸收穿过黑洞事件视界的物质，并以接近光速的速度将其余物质喷射到太空中，天体物理学家称之为‘ \
        相对论性喷流’。”比如，此次事件视界望远镜的拍照“模特”M87星系中心黑洞就因其令人印象深刻的喷 \
        射而声名显赫，它喷射的物质和辐射遍布整个太空。它的“相对论性喷流”如此庞大，以至于它们可 \
        以完全逃离周围的星系。'
cn_clf.predict(text)
```
结果为: 

```console
['科技']
```
预测同样可以用demo中的predict.py文件，以命令行方式调用。  
命令行预测：  
可以传入文本或者一个文本文件，使用哪个模型需要在configure.json中选择，只能选择一个模型。  
```console
python predict.py --text=" " --file=" "
```
#### 训练
实体识别模型：  
数据格式：  
Token1 Tag  
Token2 Tag  
...  
\n          
Token1 Tag  
...

```python
from lensnlp import seq_train
path = './ner_data'
train_file = 'train.txt'
test_file = 'test.txt'
language = 'cn'
out_path = './out'
cn_ner_train = seq_train(path,train_file,out_path,language,test_file)
cn_ner_train.train()
```
文本分类：  
数据格式：  
label text1  
label text2  
label text3  
label text4  
...  

```python
from lensnlp import cls_train
path = './clf_data'
train_file = 'train.txt'
test_file = 'test.txt'
language = 'cn'
out_path = './out'
cn_clf_train = cls_train(path,train_file,out_path,language,test_file)
cn_clf_train.train()
```
训练可用demo中的train.py调用，方式如下：  
命令行训练：  
先在configure.json中配置训练信息：  
```console
model_name:{
      "task_type":"clf",
      "train_file":"./dataset/train.txt",
      "test_file":"./dataset/test.txt",
      "language":"CN_char",
      "out_path":"./dataset"
    }
```
 然后，
 ```console
python train.py --name=""
 ```
 就开始训练。模型会储存在outpath中，模型的信息会自动加入到configure.json中。  
#### http接口：  
在configure.json文件中，将需要加载的模型的"execute"设置为true，支持同时加载多个模型。然后执行以下命令行
```console
python app.py configure.json 4000
```
其中 4000是端口，若不设置，则默认为4000。  
传入数据的格式为{"name":"model_name","data":"str or list(str)(for clf)""}。
