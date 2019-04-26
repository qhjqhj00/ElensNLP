from lensnlp.utilis.data import TaggedCorpus, Sentence, Token
from typing import List, Dict, Union
import logging

import re
from lensnlp.hyper_parameters import tag_filter
from pathlib import Path

log = logging.getLogger('lensnlp')


def load_column_corpus(
        data_folder: Union[str, Path],
        column_format: Dict[int, str],
        train_file=None,
        test_file=None,
        dev_file=None,
        tag_to_biloes=None,
        lang=None) -> TaggedCorpus:

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
        sentences_train.extend(read_column_data(path, column_format, lang))

    if test_file is not None:
        sentences_test: List[Sentence] = read_column_data(test_file, column_format, lang)

    else:
        sentences_test: List[Sentence] = [sentences_train[i] for i in
                                          __sample(len(sentences_train), 0.2)] # 自动切分测试集
        sentences_train = [x for x in sentences_train if x not in sentences_test]

    if dev_file is not None:
        sentences_dev: List[Sentence] = read_column_data(dev_file, column_format, lang)

    else:
        sentences_dev: List[Sentence] = [sentences_train[i] for i in
                                         __sample(len(sentences_train), 0.05)] # 自动切分 验证集

    if tag_to_biloes is not None:
        for sentence in sentences_train + sentences_test + sentences_dev:
            sentence: Sentence = sentence
            sentence.convert_tag_scheme(tag_type=tag_to_biloes, target_scheme='iobes')

    return TaggedCorpus(sentences_train, sentences_dev, sentences_test, name=data_folder.name)


def read_column_data(path_to_column_file: Path, column_name_map: Dict[int, str], lang):

    sentences: List[Sentence] = []
    ner_list = tag_filter  # 不在这个列表中的标签会被标称 O
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
            token = Token(fields[text_column],lang=lang)
            for column in column_name_map:
                if len(fields) > column:
                    if column != text_column:
                        if column_name_map[column] == 'ner' and fields[column] not in ner_list:
                            fields[column] = 'O'
                        token.add_tag(column_name_map[column], fields[column])

            sentence.add_token(token)

    if len(sentence.tokens) > 0:
        sentence.infer_space_after()
        sentences.append(sentence)
    sentences = [sent for sent in sentences if len(sent) < 200]
    return sentences


def load_clf_data(tag, train_file, test_file=None):

    train_data = open(str(train_file), encoding='utf-8').read().strip().split('\n')

    train_data = [re.split("\s+", doc) for doc in train_data]
    train_y = [doc[0] for doc in train_data]
        if tag in ['CN_char', 'CN_token']:
                train_X = [doc[1].replace(' ','') for doc in train_data]
        elif tag == 'EN':
                train_X = [' '.join(doc[1:]) for doc in train_data]
    train_ = [Sentence(train_X[i], tag, [train_y[i]]) for i in range(len(train_X)) if len(train_X[i]) > 0]

    import random
    random.shuffle(train_)

    if test_file is not None:
        test_data = open(str(test_file), encoding='utf-8').read().strip().split('\n')
        test_data = [doc.split('\t') for doc in test_data]
        test_y = [doc[0] for doc in test_data]
#         test_X = [doc[1].replace(' ','') for doc in test_data]
        if tag in ['CN_char', 'CN_token']:
            test_X = [doc[1].replace(' ', '') for doc in train_data]
        elif tag == 'EN':
            test_X = [' '.join(doc[1:]) for doc in train_data]
        test_ = [Sentence(test_X[i], tag, [test_y[i]]) for i in range(len(test_X)) if len(test_X[i]) > 0]
    else:
        test_: List[Sentence] = [train_[i] for i in
                                __sample(len(train_), 0.2)]
        train_ = [sentence for sentence in train_ if sentence not in test_]

    dev_: List[Sentence] = [train_[i] for i in
                                 __sample(len(train_), 0.2)]

    corpus = TaggedCorpus(train_, dev_, test_)

    return corpus


def __sample(total_number_of_sentences: int, percentage: float = 0.1) -> List[int]:
    import random
    sample_size: int = round(total_number_of_sentences * percentage)
    sample = random.sample(range(1, total_number_of_sentences), sample_size)
    return sample

