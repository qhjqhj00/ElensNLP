### 利用lensnlp 进行模型训练

---

#### 词向量

Lensnlp中内置了多种预训练语言模型。在安装lensnlp后，会在用户目录下生成 .lensnlp/ 文件夹，所有的预训练语言模型都需要放置在 .lensnlp/language_model/中。

1. Word2Vec，Glove，Fasttext类型语言模型：

   ```python
   from from lensnlp.Embeddings import WordEmbeddings
   from lensnlp.utilis.data import Sentence
   Word_embed = WordEmbeddings('cn_glove')
   sent = Sentence('北京一览群智数据有限公司。')
   Word_embed.embed((sent))
   ```
|语言模型|标签|语种|
|:--:|:---:|:---:|
|Glove|cn_glove|中文|
|Glove|en_glove|英文|
|Fasttext|cn_fasttext|中文|


2. BPemb语言模型

   BPemb相关信息可参照：
   
   https://github.com/bheinzerling/bpemb
   
   ```python
   from lensnlp.Embeddings import BytePairEmbeddings
   Word_embed = BytePairEmbeddings('ug',300,50000)
   ```
   
   
   
3. Bert，Flair 等根据上下文生成的语言模型：

   ```python
   from lensnlp.Embeddings import FlairEmbeddings
   Word_embed = FlairEmbeddings('cn_forward_small')
   
   from lensnlp.Embeddings import BertEmbeddings
   Word_embed = BertEmbeddings('chinese-base-uncased')
   ```

   Bert模型依赖于pytorch_pretrained_bert，模型名称可参考：

   https://github.com/huggingface/pytorch-pretrained-BERT

   如果预训练模型下载失败，可以手动下载，然后修改pytorch_pretrained_bert中model.py 和 tokenizer.py中对应的模型地址。

   Flair字符级语言模型相关信息可参考：

   https://github.com/zalandoresearch/flair

   lensnlp中预置的Flair语言模型中，英语模型是作者训练，中文和维吾尔语模式是Lensnlp自己训练的。
|模型标签|hidden size|语种|方向|
|:---:|:--:|:--:|:--:|
|cn_forward_large|2048|中文|正向|
|cn_backward_large|2048|中文|反向|
|cn_forward_small|1024|中文|正向|
|cn_backward_small|1024|中文|反向|
|en_forward_large|2048|英文|正向|
|en_backward_large|2048|英文|反向|
|en_forward_small|1024|英文|正向|
|en_backward_small|1024|英文|反向|
|uy_forward_large|2048|维文|正向|
|uy_backward_large|2048|维文|反向|
|uy_forward_small|1024|维文|正向|
|uy_backward_small|1024|维文|反向|

4. 使用多种语言模型

   如果想叠加多种语言模型，可以用StackedEmbeddings类。

   ```python
   from lensnlp.Embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
   Embed_list = [WordEmbeddings('cn_glove'),FlairEmbeddings('cn_forward_small')]
   Word_embed = StackedEmbeddings(Embed_list)
   ```

   

#### 序列标注模型

lensnlp提供 BiLSTM+CRF结构的序列标注模型。其中，RNN类网络和是否使用CRF都是可调的。

```python
from lensnlp.utilis.data import TaggedCorpus
from lensnlp.utilis.data_load import load_column_corpus
from lensnlp.Embeddings import *

# 数据集每一行的类型，CONLL03 数据集为 {0: 'text', 1: 'pos', 2: 'chunk', 3: 'chunk'}
columns = {0: 'text', 1: 'ner'}  
data_folder = '.'  # 数据集的路径
corpus: TaggedCorpus = load_column_corpus(data_folder, columns, train_file,
                                         test_file,language)

tag_type = 'ner'  # 序列类型

tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)  # 标签词典

Embed_list = [WordEmbeddings('cn_glove'),FlairEmbeddings('cn_forward_small')]

from lensnlp.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,  # 初始化模型
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

from lensnlp.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)  # 初始化训练，参数推荐默认值

trainer.train('cn/ner/',  # 开始训练
            	learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150)
```



#### 文本分类和情感分析模型

1. 目前文本分类和情感分析模型使用 TextClassifier 类，该类中只有一层全连结层进行Decoding，之前的操作都在doc_embeddings中操作。

   ```python
   from lensnlp.Embeddings import *
   Embed_list = [WordEmbeddings('cn_glove')]
   
   # 1.全局平均值
   Docum_embed = DocumentPoolEmbeddings(Embed_list)
   
   # 2. RNN类型，可选择RNN种类，是否双向，是否加注意力
   Docum_embed = DocumentRNNEmbeddings(Embed_list,bidirectional=True,use_attention=True)
   
   # 3. CNN1D
   Docum_embed = DocumentCNN1DEmbedding(Embed_list)
   
   # 4. CNN2D
   DocumentCNN2DEmbedding(Embed_list)
   
   # 5. 利用Flair语言模型
   Embed_list = [FlairEmbeddings('cn_forward_small'),FlairEmbeddings('cn_backward_small')]
   Docum_embed = DocumentLMEmbeddings(Embed_list)
   
   # 6. RCNN
   Docum_embed = DocmentRCNNEmbedding(Embed_list)
   ```

   

2. 训练文本分类和情感分析的分类器是一样的，代码如下：

   ```python
   from lensnlp.utilis.data import TaggedCorpus
   from lensnlp.utilis.data_load import load_clf_data
   from lensnlp.Embeddings import *
   
   # 语言选择：CN_char:按字符分中文；CN_token:中文分词；EN:英文；UY:维吾尔语
   corpus = load_clf_data('CN_char',train_data,test_data) 
   
   label_dictionary = corpus.make_label_dictionary()
   
   embed_list = [WordEmbeddings('cn_glove')]
   
   doc_embed = DocumentRNNEmbeddings(embed_list)
   
   from lensnlp.models import TextClassifier
   
   clf: TextClassifier = TextClassifier(doc_embed, label_dictionary, multi_label=False)
     
   from lensnlp.trainers import ModelTrainer
   
   trainer: ModelTrainer = ModelTrainer(clf, corpus)
     
   trainer.train('cn/clf/',
                 learning_rate=0.1,
                 mini_batch_size=32,
                 max_epochs=150)
   ```

   