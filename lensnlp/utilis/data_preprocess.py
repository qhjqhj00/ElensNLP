from lensnlp.utilis.data import Sentence
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from typing import List,Union
import pymongo
import requests


def cn_prepare(text: str) -> List[Sentence]:
    """中文分句"""
    sentences: List[Sentence] = []
    text = re.sub('([。！？\?])([^”’])', r"\1\n\2", text)
    text = re.sub('(\.{6})([^”’])', r"\1\n\2", text)
    text = re.sub('(\…{2})([^”’])', r"\1\n\2", text)
    text = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', text)
    text = text.split('\n')
    for s in text:
        if len(s) > 500:
            n = int(len(s)/500)+1
            tem = [Sentence(s[i*500:(i+1)*500],'PY') for i in range(n)]
            sentences.extend(tem)
        else:
            sentences.append(Sentence(s,'PY'))
    return sentences


def en_prepare(text: str) -> List[Sentence]:
    """英文分句"""
    sentences: List[Sentence] = []
    text = sent_tokenize(text)
    text = [word_tokenize(s) for s in text]
    text = [' '.join(s) for s in text]
    for s in text:
        sentences.append(Sentence(s,'EN'))
    return sentences


def uy_prepare(text:str) -> List[Sentence]:
    """维吾尔语分句"""
    text = re.sub('\.', '.<SPLIT>', text)
    text = re.sub('!', '!<SPLIT> ', text)
    text = re.sub('؟', '؟<SPLIT>', text)
    text = text.split('<SPLIT>')
    sentences: List[Sentence] = []
    for s in text:
        if len(s)>0:
            sentences.append(Sentence(s,'UY'))
    return sentences


def clf_preprocess(text:Union[str,list],language):
    """文本分类数据分句"""
    if type(text) is str:
        text = [text]
    if language == 'CN_char':
        sentences: List[Sentence] = [(Sentence(s[:2000],'CN_char')) for s in text]
    elif language == 'CN_token':
        sentences: List[Sentence] = [(Sentence(s[:2000], 'CN_token')) for s in text]
    elif language == 'UY':
        sentences: List[Sentence] = [(Sentence(s, 'UY')[:2000]) for s in text]
    elif language == 'EN':
        sentences: List[Sentence] = [(Sentence(s, 'EN')[:2000]) for s in text]
    else:
        raise ValueError('Not Yet!')
    return sentences


def repl_str(matched):
    replace_str = matched.group('symbol')
    return ' ' + replace_str + " "


def uy_segmentation(word):

    end_shape_affix_list = ['دىكىلەرنىڭ', 'لىقلاردىن', 'لىقلارنىڭ', 'لىقلاردەك', 'لۇقلاركەن', 'لىكلەرنىڭ',
                                'لىقلارغا', 'لىقلاردا', 'لىرىنىڭ', 'لىرىنىڭ', 'لىقلار', 'لىكلەر', 'لىكتىن', 'چىلىك',
                                'لىقنى', 'لەردە', 'لاردا', 'لىرىم', 'دىنمۇ', 'دىكى', 'تىكى', 'غىچە', 'قىچە', 'گىچە',
                                'كىچە', 'لىرى', 'نىڭ', 'دىن', 'تىن', 'دەك', 'تەك', 'لار', 'لەر', 'لىق', 'لىك', 'لىك',
                                'نى', 'دا', 'دە', 'تا', 'تە', 'چە', 'غا', 'قا', 'گە', 'كە', 'ﯘم', 'دا', 'دە', 'مۇ']

    drop_end = re.sub('(?P<symbol>' + '|'.join([x + '$' for x in end_shape_affix_list]) + ')', repl_str, word, count=1)
    end = ''
    if len(drop_end.split()) > 1:
        end = drop_end.split()[1]
        drop_end = drop_end.split()[0]

    return [drop_end, end]





