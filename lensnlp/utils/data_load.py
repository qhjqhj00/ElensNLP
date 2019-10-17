from lensnlp.utils.data import TaggedCorpus, Sentence, Token, Label
from typing import List, Dict, Union
import logging

import re
from lensnlp.hyper_parameters import Parameter
from pathlib import Path
from segtok.tokenizer import word_tokenizer

from lensnlp.utils.data_preprocess import re_clean_str

log = logging.getLogger('lensnlp')


def load_column_corpus(
        data_folder: Union[str, Path],
        column_format: Dict[int, str],
        train_file=None,
        test_file=None,
        dev_file=None,
        tag_to_bioes=None,
        sp=None,
        max_length=1024) -> TaggedCorpus:
    """
    加载标准的序列标注数据
    格式例如：
    Beijing B-LOC

    is O

    a O

    foggy O

    city O

    where O

    Elensdata B-ORG

    locates O

    . O


    :param data_folder: 数据文件夹路径
    :param column_format: 数据格式，两列ner为 {1:'token',2:'ner'}
    :param train_file: 训练数据文件名
    :param test_file:  测试数据文件名
    :param dev_file:  验证数据文件名
    :param tag_to_biloes: 是否换为bioes标签策略
    :param lang: 语种 中文：CN_char（按字符分）CN_token（分词）维吾尔语：UY
    :return: corpus
    Examples
    --------
    >>> from lensnlp.utils.data_load import load_column_corpus
    >>> corpus = load_column_corpus('./dataset/',{1:'token',2:'ner'},'train.txt','test.txt',lang='UY')
    """

    if type(data_folder) == str:
        data_folder: Path = Path(data_folder)
    train_ = []
    if train_file is not None:
        for tr in train_file:
            train_.append(data_folder / tr)
    if test_file is not None:
        test_file = data_folder / test_file
    if dev_file is not None:
        dev_file = data_folder / dev_file

    log.info("Reading data from {}".format(data_folder))
    for path in train_:
        log.info("Train: {}".format(path))
    log.info("Dev: {}".format(dev_file))
    log.info("Test: {}".format(test_file))
    sentences_train: List[Sentence] = []

    for path in train_:
        sentences_train.extend(read_column_data(path, column_format, sp=sp, max_length=max_length))

    if test_file is not None:
        sentences_test: List[Sentence] = read_column_data(test_file, column_format, sp=sp, max_length=max_length)

    else:
        sentences_test: List[Sentence] = [sentences_train[i] for i in
                                          __sample(len(sentences_train), 0.2)] # 自动切分测试集
        sentences_train = [x for x in sentences_train if x not in sentences_test]

    if dev_file is not None:
        sentences_dev: List[Sentence] = read_column_data(dev_file, column_format, sp=sp, max_length=max_length)

    else:
        sentences_dev: List[Sentence] = [sentences_train[i] for i in
                                         __sample(len(sentences_train), 0.05)] # 自动切分 验证集
        sentences_train = [x for x in sentences_train if x not in sentences_dev]

    if tag_to_bioes is not None:
        for sentence in sentences_train + sentences_test + sentences_dev:
            sentence: Sentence = sentence
            sentence.convert_tag_scheme(tag_type=tag_to_bioes, target_scheme='iobes')

    return TaggedCorpus(sentences_train, sentences_dev, sentences_test, name=data_folder.name)


def read_column_data(path_to_column_file: Path, column_name_map: Dict[int, str], sp=None, max_length=1024):
    """
    :param path_to_column_file: 数据文件夹路径
    :param column_name_map: 数据格式，两列ner为 {1:'token',2:'ner'}
    :param lang: 语种 中文：CN_char（按字符分）CN_token（分词）维吾尔语：UY
    :return: corpus
    """

    sentences: List[Sentence] = []
    # ner_list = tag_filter  # 不在这个列表中的标签会被标称 O
    try:
        lines: List[str] = open(str(path_to_column_file), encoding='utf-8').read().strip().split('\n')
    except:
        log.info('UTF-8 can\'t read: {} ... using "latin-1" instead.'.format(path_to_column_file))
        lines: List[str] = open(str(path_to_column_file), encoding='latin1').read().strip().split('\n')

    text_column: int = 0
    for column in column_name_map:
        if column_name_map[column] == 'text':
            text_column = column

    sentence: Sentence = Sentence()
    for line in lines:

        if line.startswith('#'):
            continue

        if line.strip().replace('﻿', '') == '':
            if len(sentence) > 0:
                sentence.infer_space_after()
                sentences.append(sentence)
            sentence: Sentence = Sentence()

        else:
            fields: List[str] = re.split("\s+", line)
            if len(fields[text_column]) == 0:
                continue
            token = Token(fields[text_column], sp=sp)
            for column in column_name_map:
                if len(fields) > column:
                    if column != text_column:
                        token.add_tag(column_name_map[column], fields[column])

            sentence.add_token(token)

    if len(sentence.tokens) > 0:
        sentence.infer_space_after()
    sentences.append(sentence)
    sentences = [sentence for sentence in sentences if len(sentence) < max_length]
    return sentences


def load_clf_data(language, train_file, test_file=None, max_length: int = 1024, sp_op = None):
    """
    加载文本分类的数据
    :param tag: 语种 中文：CN_char（按字符分）CN_token（分词）维吾尔语：UY
    :param train_file: 训练数据文件路径
    :param test_file: 测试数据文件路径
    :param max_length
    :return: corpus
    """

    train_data = open(str(train_file), encoding='utf-8',errors='ignore').read().strip().split('\n')

    train_data = [re.split("\t", doc) for doc in train_data]
    train_data = [doc for doc in train_data if len(doc) == 2]
    train_y = [doc[0] for doc in train_data]
    if language == 'zh':
            train_X = [doc[1].replace(' ','') for doc in train_data]
    else:
            train_X = [doc[1] for doc in train_data]
    train_ = [Sentence(train_X[i], language_type=language, labels=[train_y[i]], max_length=max_length, sp_op=sp_op)
              for i in range(len(train_X)) if len(train_X[i]) > 0]

    import random
    random.shuffle(train_)

    if test_file is not None:
        test_data = open(str(test_file), encoding='utf-8',errors='ignore').read().strip().split('\n')
        test_data = [doc.split('\t') for doc in test_data]
        test_data = [doc for doc in test_data if len(doc) == 2]
        test_y = [doc[0] for doc in test_data]
#         test_X = [doc[1].replace(' ','') for doc in test_data]
        if language == 'zh':
            test_X = [doc[1].replace(' ', '') for doc in test_data]
        else:
            test_X = [doc[1] for doc in test_data]
        test_ = [Sentence(test_X[i], language_type=language, labels=[test_y[i]], max_length=max_length, sp_op=sp_op)
                 for i in range(len(test_X)) if len(test_X[i]) > 0]
    else:
        test_: List[Sentence] = [train_[i] for i in
                                __sample(len(train_), 0.2)]
        train_ = [sentence for sentence in train_ if sentence not in test_]

    dev_: List[Sentence] = [train_[i] for i in
                                 __sample(len(train_), 0.2)]

    corpus = TaggedCorpus(train_, dev_, test_)

    return corpus


def __sample(total_number_of_sentences: int, percentage: float = 0.1) -> List[int]:
    """
    :param total_number_of_sentences: 数据集的大小
    :param percentage: 提取比例
    :return: sample的indices
    """
    import random
    sample_size: int = round(total_number_of_sentences * percentage)
    sample = random.sample(range(1, total_number_of_sentences), sample_size)
    return sample


def load_re_data(train_file, test_file: str = None):
    train_ = read_re_data(train_file)
    import random
    random.shuffle(train_)

    if test_file is not None:
        test_ = read_re_data(test_file)
    else:
        test_: List[Sentence] = [train_[i] for i in
                                __sample(len(train_), 0.2)]
        train_ = [sentence for sentence in train_ if sentence not in test_]

    dev_: List[Sentence] = [train_[i] for i in
                                 __sample(len(train_), 0.2)]

    corpus = TaggedCorpus(train_, dev_, test_)
    return corpus


def read_re_data(path):
    """
    :param path:path of file
    :param tag: language of file
    :return:
    """
    data = []
    lines = [line.strip() for line in open(path)]
    for idx in range(0, len(lines), 4):
        sent = Sentence()
        id = lines[idx].split("\t")[0]
        relation = lines[idx + 1]

        sentence = lines[idx].split("\t")[1][1:-1]
        sentence = sentence.replace('<e1>', ' _e11_ ')
        sentence = sentence.replace('</e1>', ' _e12_ ')
        sentence = sentence.replace('<e2>', ' _e21_ ')
        sentence = sentence.replace('</e2>', ' _e22_ ')

        sentence = re_clean_str(sentence)
        tokens = word_tokenizer(sentence)

        for i,token in enumerate(tokens):

            if token == 'e12':
                sent.entity['e1'] = str(i-1)
            elif token == 'e22':
                sent.entity['e2'] = str(i-1)
            sent.add_token(Token(token))

        sent.add_label(Label(relation))

        data.append(sent)
    for sent in data:
        sent.generate_relative_pos(Parameter['re_max_length'])
    return data
