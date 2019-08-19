from lensnlp.utils.data_load import load_column_corpus
from lensnlp.embeddings import WordEmbeddings, DocumentPoolEmbeddings

from torch.optim.adam import Adam
from lensnlp.utils.data_load import load_clf_data

from pathlib import Path


class seq_train:
    """序列标注训练

            Parameters
            ----------

            path : str
                训练文件的路径
            train_file : str
                训练数据文件名
            out_path : str
                输出路径
            language : str
                 数据语言类型
            test_file : str
                测试数据文件名
            Examples
            --------
             >>> from lensnlp import seq_train
             >>> path = './ner_data'
             >>> train_file = 'train.txt'
             >>> test_file = 'test.txt'
             >>> language = 'cn'
             >>> out_path = './out'
             >>> cn_ner_train = seq_train(path,train_file,out_path,language,test_file)

            """

    def __init__(self,path,train_file,out_path,language,test_file=None):

        self.corpus = load_column_corpus(path, {0: 'text', 1: 'ner'}, train_file=[train_file],test_file = test_file)
        self.tag_type = 'ner'
        self.language = language
        self.outpath = out_path
        self.tag_dict = self.corpus.make_tag_dictionary(tag_type=self.tag_type)

    def train(self):
        if self.language == 'cn':
            embeddings = WordEmbeddings('cn_glove')
        elif self.language == 'en':
            embeddings = WordEmbeddings('en_glove')
        else:
            raise ValueError('Not yet!')
        print('Embeddings prepared...')
        from lensnlp.models import SequenceTagger

        tagger: SequenceTagger = SequenceTagger(hidden_size=256,  # 初始化模型
                                                embeddings=embeddings,
                                                tag_dictionary=self.tag_dict,
                                                tag_type=self.tag_type,
                                                use_crf=True)
        print('Tagger prepared...')
        from lensnlp.trainers import ModelTrainer

        trainer: ModelTrainer = ModelTrainer(tagger, self.corpus)
        print('Trainer prepared...')
        print('Begin to train...')
        trainer.train(self.outpath,  # 开始训练
                      learning_rate=0.1,
                      mini_batch_size=32,
                      max_epochs=150)


class cls_train:
    """实体识别模型，提供中文，英文，维吾尔语三个语种的预训练模型。

            Parameters
            ----------

            path : str
                训练文件的路径
            train_file : str
                训练数据文件名
            out_path : str
                输出路径
            language : str
                 数据语言类型
            test_file : str
                测试数据文件名
            Examples
            --------
             >>> from lensnlp import cls_train
             >>> path = './ner_data'
             >>> train_file = 'train.txt'
             >>> test_file = 'test.txt'
             >>> language = 'cn'
             >>> out_path = './out'
             >>> cn_ner_train = cls_train(path,train_file,out_path,language,test_file)

            """

    def __init__(self, path, train_file, out_path, language, test_file=None):
        if language == 'cn':
            if test_file is not None:
                self.corpus = load_clf_data('CN_char',Path(path) / train_file, Path(path) / test_file)
            else:
                self.corpus = load_clf_data('CN_char', Path(path) / train_file)
            self.embedding_list = [WordEmbeddings('cn_glove')]
        elif language == 'en':
            if test_file is not None:
                self.corpus = load_clf_data('EN',Path(path) / train_file, Path(path) / test_file)
            else:
                self.corpus = load_clf_data('EN', Path(path) / train_file)
            self.embedding_list = [WordEmbeddings('en_glove')]
        else:
            raise ValueError('Not yet!')

        self.label_dictionary = self.corpus.make_label_dictionary()
        self.outpath = out_path

    def train(self):
        doc_embed = DocumentPoolEmbeddings(self.embedding_list)

        from lensnlp.models import TextClassifier

        clf: TextClassifier = TextClassifier(doc_embed,self.label_dictionary,multi_label=False) # 初始化模型

        from lensnlp.trainers import ModelTrainer

        trainer: ModelTrainer = ModelTrainer(clf, self.corpus, optimizer=Adam) # 训练，参数推荐默认值

        trainer.train(self.outpath,
                learning_rate=0.001,
                 mini_batch_size=32,
                 max_epochs=150)

