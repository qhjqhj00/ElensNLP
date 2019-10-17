from lensnlp.utils.data import Sentence
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from typing import List,Union
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
            tem = [Sentence(s[i*500:(i+1)*500], language_type='zh', sp_op='py4c') for i in range(n)]
            sentences.extend(tem)
        else:
            sentences.append(Sentence(s, language_type='zh', sp_op='py4c'))
    return sentences


def en_prepare(text: str) -> List[Sentence]:
    """英文分句"""
    sentences: List[Sentence] = []
    text = sent_tokenize(text)
    text = [word_tokenize(s) for s in text]
    text = [' '.join(s) for s in text]
    for s in text:
        sentences.append(Sentence(s))
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
            sentences.append(Sentence(s, language_type='ug', sp_op='lt'))
    return sentences


def clf_preprocess(text:Union[str,list],language, max_length: int = 1024):
    """文本分类数据分句"""
    if type(text) is str:
        text = [text]
    if language == 'CN_char':
        sentences: List[Sentence] = [(Sentence(s,language_type='zh', sp_op='char', max_length=max_length)) for s in text]
    elif language == 'CN_token':
        sentences: List[Sentence] = [(Sentence(s,language_type='zh', max_length=max_length)) for s in text]
    elif language == 'UY':
        sentences: List[Sentence] = [(Sentence(s,language_type='ug', max_length=max_length)) for s in text]
    elif language == 'EN':
        sentences: List[Sentence] = [(Sentence(s, language_type='en', max_length=max_length)) for s in text]
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


def regx_ner(text):
    regex_dict = {'MAIL':'[0-9a-zA-Z_]{0,19}@[0-9a-zA-Z\.]{1,18}\.[a-z]{1,6}',
              'TELE':'\(?0\d{2,3}[)-]?\d{7,8}|(?:(\+\d{2}))?(\d{1,3})\d{8}',
              'WEB':'([a-zA-Z]+://)?([a-zA-Z0-9._-]{1,66})\.[a-z]{1,5}(:[0-9]{1,4})*(/[a-zA-Z0-9&%_./-~-]*)*',
              'TERM':'《([^》]*){1,20}》',
              'TAG':'#([^#]*){1,32}#'}
    text=text.replace('．','.')
    text=text.replace('－','-')
    result = []
    for key in regex_dict:
        res = re.finditer(regex_dict[key],text)
        result.extend([{'text':t.group(),'start_pos':t.start(),
                   'end_pos':t.end(),'type':key} for t in res])
    return result


def re_clean_str(text):
    text = text.lower()
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()



