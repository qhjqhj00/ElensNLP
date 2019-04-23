from lensnlp.utilis.data import Sentence
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from typing import List,Union


def cn_prepare(text: str) -> List[Sentence]:
    '''Split Chinese text and return sentence list'''
    sentences: List[Sentence] = []
    text = re.sub('([。！？\?])([^”’])', r"\1\n\2", text)
    text = re.sub('(\.{6})([^”’])', r"\1\n\2", text)
    text = re.sub('(\…{2})([^”’])', r"\1\n\2", text)
    text = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', text)
    text = text.split('\n')
    for s in text:
        sentences.append(Sentence(s,'PY'))
    return sentences


def en_prepare(text: str) -> List[Sentence]:
    '''Split English text and return sentence list'''
    sentences: List[Sentence] = []
    text = sent_tokenize(text)
    text = [word_tokenize(s) for s in text]
    text = [' '.join(s) for s in text]
    for s in text:
        sentences.append(Sentence(s,'EN'))
    return sentences


def uy_prepare(text:str) -> List[Sentence]:
    text = re.sub('\.', '.<SPLIT>', text)
    text = re.sub('!', '!<SPLIT> ', text)
    text = re.sub('؟', '؟<SPLIT>', text)
    text = text.split('<SPLIT>')
    sentences: List[Sentence] = []
    for s in text:
        if len(s)>0:
            sentences.append(Sentence(s,'UY'))
    return sentences


def clf_preprocess(text:Union[str,list]):
    if type(text) is str:
        text = [text]
    sentences: List[Sentence] = [(Sentence(s[:500],'CN_char')) for s in text]
    return sentences



