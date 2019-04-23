from lensnlp.utilis.data import TaggedCorpus
from lensnlp.utilis.data_load import load_column_corpus, load_clf_data
from lensnlp.Embeddings import *

import argparse
import json
import os

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--configure', type=str,default='configure.json')

args = parser.parse_args()

CACHE_ROOT = os.path.expanduser(os.path.join('~', '.lensnlp'))

with open(args.configure,'r') as f:
    all_configure = json.load(f)

configure = all_configure['train'][args.name]
all_configure['predict'][args.name] = {
        "execute": False,
        "language": configure['language'],
        "type": configure['task_type'],
        "model_path": configure['out_path']+"/best-model.pt",}

with open(args.configure,'w') as f:
    json.dump(f, all_configure)

if configure['task_type'] == 'ner':

    columns = {0: 'text', 1: 'ner'}  # 数据集每一行的类型，CONLL03 数据集为 {0: 'text', 1: 'pos', 2: 'chunk', 3: 'chunk'}
    data_folder = '.'  # 数据集的路径

    # 为了方便多数据集合并，支持多文件传入，如 ['data_1.txt','data_2.txt','data_3.txt']
    corpus: TaggedCorpus = load_column_corpus(data_folder, columns, train_file=[configure['train_file']],
                                              test_file=configure['test_file'],lang=configure['language'])

    tag_type = 'ner'  # 序列类型

    print(corpus.obtain_statistics(tag_type))

    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)  # 标签词典

    print(tag_dictionary.idx2item)

    embedding_types = []
    if configure['language'] == 'CN_char':
        embedding_types.append(WordEmbeddings('cn_glove'))
    elif configure['language'] == 'EN':
        embedding_types.append(WordEmbeddings('en_glove'))
    else:
        raise ValueError('Not Yet!')

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)  # 将list中的词向量依次加入

    from lensnlp.models import SequenceTagger

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,  # 初始化模型
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True)

    from lensnlp.trainers import ModelTrainer

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)  # 初始化训练，参数推荐默认值

    trainer.train(configure['out_path'],  # 开始训练
                  learning_rate=0.1,
                  mini_batch_size=32,
                  max_epochs=150)

elif configure['task_type'] == 'clf':

    from torch.optim.adam import Adam

    corpus = load_clf_data('CN_char',configure['train_file'],
                           configure['test_file'])  # 加载数据
    label_dictionary = corpus.make_label_dictionary()  # 获得标签字典
    print(len(corpus.train))
    print(len(corpus.test))

    embed_list = [WordEmbeddings('cn_glove')]

    doc_embed = DocumentRNNEmbeddings(embed_list)  # 均值计算，可选 mode = ['min','max',mean']

    from lensnlp.models import TextClassifier

    clf: TextClassifier = TextClassifier(doc_embed, label_dictionary, multi_label=False)  # 初始化模型

    from lensnlp.trainers import ModelTrainer

    trainer: ModelTrainer = ModelTrainer(clf, corpus, optimizer=Adam)  # 训练，参数推荐默认值

    trainer.train(configure['out_path'],
                  learning_rate=0.001,
                  mini_batch_size=32,
                  max_epochs=150)

else:
    raise ValueError('暂不支持该功能！')
