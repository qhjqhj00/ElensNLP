from abc import abstractmethod
from typing import List, Dict, Union

import torch
from pypinyin import pinyin, Style
from collections import Counter
from collections import defaultdict
from bpemb import BPEmb
import jieba

from segtok.segmenter import split_single
from segtok.tokenizer import split_contractions
from segtok.tokenizer import word_tokenizer

import re


class Dictionary:

    """
    字典类
    """

    def __init__(self, add_unk=True):
        self.item2idx: Dict[str, int] = {}
        self.idx2item: List[str] = []
        if add_unk:
            self.add_item('<unk>')

    def add_item(self, item: str) -> int:

        item = item.encode('utf-8')
        if item not in self.item2idx:
            self.idx2item.append(item)
            self.item2idx[item] = len(self.idx2item) - 1
        return self.item2idx[item]

    def get_idx_for_item(self, item: str) -> int:

        item = item.encode('utf-8')
        if item in self.item2idx.keys():
            return self.item2idx[item]
        else:
            return 0

    def get_items(self) -> List[str]:
        items = []
        for item in self.idx2item:
            items.append(item.decode('UTF-8'))
        return items

    def __len__(self) -> int:
        return len(self.idx2item)

    def get_item_for_index(self, idx):
        return self.idx2item[idx].decode('UTF-8')

    def save(self, savefile):
        import pickle
        with open(savefile, 'wb') as f:
            mappings = {
                'idx2item': self.idx2item,
                'item2idx': self.item2idx
            }
            pickle.dump(mappings, f)

    @classmethod
    def load(cls, filename: str):
        import pickle
        dictionary: Dictionary = Dictionary()
        with open(filename, 'rb') as f:
            mappings = pickle.load(f, encoding='latin1')
            idx2item = mappings['idx2item']
            item2idx = mappings['item2idx']
            dictionary.item2idx = item2idx
            dictionary.idx2item = idx2item
        return dictionary


class Label:

    """
    标签类，包括序列标签和文本标签；
    value是标签的文本，score是标签的得分
    """

    def __init__(self, value: str, score: float = 1.0):
        self.value = value
        self.score = score
        super().__init__()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if not value and value != '':
            raise ValueError('Incorrect label value provided. Label value needs to be set.')
        else:
            self._value = value

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, score):
        if 0.0 <= score <= 1.0:
            self._score = score
        else:
            self._score = 1.0

    def to_dict(self):
        return {
            'value': self.value,
            'confidence': self.score
        }

    def __str__(self):
        return "{} ({})".format(self._value, self._score)

    def __repr__(self):
        return "{} ({})".format(self._value, self._score)


class Token:
    """
    标签类
    目前针对中文，英文，维吾尔语做了区别处理。
    其中，参数lang控制不同的语言
    lang = 'UY' 维吾尔语生成 Token.latin 拉丁字母
    lang = 'PY' 中文生成 Token.pinyin 拼音
    """

    def __init__(self,
                 text: str,
                 idx: int = None,
                 head_id: int = None,
                 whitespace_after: bool = True,
                 start_position: int = None,
                 lang = None
                 ):
        self.text: str = text
        if lang == 'UY':
            self.latin = self.uyghur_to_latin()
        if lang == 'PY':
            self.pinyin = self.converter()
        self.idx: int = idx
        self.head_id: int = head_id
        self.whitespace_after: bool = whitespace_after

        self.start_pos = start_position
        self.end_pos = start_position + len(text) if start_position is not None else None

        self.sentence: Sentence = None
        self._embeddings: Dict = {}
        self.tags: Dict[str, Label] = {}
            
    def converter(self):
        p = pinyin(self.text, style=Style.TONE2)
        p = [t[0] for t in p]
        return ' '.join(p)

    def add_tag_label(self, tag_type: str, tag: Label):
        self.tags[tag_type] = tag

    def add_tag(self, tag_type: str, tag_value: str, confidence=1.0):
        tag = Label(tag_value, confidence)
        self.tags[tag_type] = tag

    def uyghur_to_latin(self):
        latin_map = {"ا": "a", "ە": "e", "ى": "i", "ې": "é", "و": "o",
                     "ۇ": "u", "ۆ": "ö", "ۈ": "ü", "ب": "b", "پ": "p", "ت": "t", "ژ": "j",
                     "چ": "ç", "خ": "x", "د": "d", "ر": "r", "ز": "z", "ج": "j", "س": "s",
                     "ش": "ş", "ف": "f", "غ": "ğ", "ق": "q", "ك": "k", "گ": "g", "ڭ": "ñ",
                     "ل": "l", "م": "m", "ن": "n", "ھ": "h", "ي": "y", "ۋ": "w", "ئ": "", ".": ".",
                     "؟": "?", "!": "!", "،": ",", "؛": ";", ":": ":", "«": "«", "»": "»",
                     "-": "-", "—": "—", "(": "(", ")": ")"}
        new = ''
        for c in self.text:
            if c in latin_map:
                new += latin_map[c]
            else:
                new += c
        return new

    def get_tag(self, tag_type: str) -> Label:
        if tag_type in self.tags: return self.tags[tag_type]
        return Label('')

    def get_head(self):
        return self.sentence.get_token(self.head_id)

    def set_embedding(self, name: str, vector: torch.autograd.Variable): # 加入向量
        self._embeddings[name] = vector.cpu()

    def clear_embeddings(self): # 清除向量
        self._embeddings: Dict = {}

    def get_embedding(self) -> torch.FloatTensor: # 获得向量

        embeddings = [self._embeddings[embed] for embed in sorted(self._embeddings.keys())]

        if embeddings:
            return torch.cat(embeddings, dim=0)

        return torch.FloatTensor()

    @property
    def start_position(self) -> int:
        return self.start_pos

    @property
    def end_position(self) -> int:
        return self.end_pos

    @property
    def embedding(self):
        return self.get_embedding()

    def __str__(self) -> str:
        return 'Token: {} {}'.format(self.idx, self.text) if self.idx is not None else 'Token: {}'.format(self.text)

    def __repr__(self) -> str:
        return 'Token: {} {}'.format(self.idx, self.text) if self.idx is not None else 'Token: {}'.format(self.text)


class Span:
    """
    实体类
    """

    def __init__(self, tokens: List[Token], tag: str = None, score=1.):
        self.tokens = tokens
        self.tag = tag
        self.score = score
        self.start_pos = None
        self.end_pos = None

        if tokens:
            self.start_pos = tokens[0].start_position
            self.end_pos = tokens[len(tokens) - 1].end_position

    @property
    def text(self) -> str:
        return ' '.join([t.text for t in self.tokens])

    def to_original_text(self) -> str:
        str = ''
        pos = self.tokens[0].start_pos
        for t in self.tokens:
            while t.start_pos != pos:
                str += ' '
                pos += 1

            str += t.text
            pos += len(t.text)

        return str

    def to_dict(self):
        return {
            'text': self.to_original_text(),
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'type': self.tag,
            'confidence': self.score
        }

    def __str__(self) -> str:
        ids = ','.join([str(t.idx) for t in self.tokens])
        return '{}-span [{}]: "{}"'.format(self.tag, ids, self.text) \
            if self.tag is not None else 'span [{}]: "{}"'.format(ids, self.text)

    def __repr__(self) -> str:
        ids = ','.join([str(t.idx) for t in self.tokens])
        return '<{}-span ({}): "{}">'.format(self.tag, ids, self.text) \
            if self.tag is not None else '<span ({}): "{}">'.format(ids, self.text)


class Sentence:

    """
    句子类，针对中文，英文，维语做了不同的分词。
    其中，中文额外提供 bpemb分词方法，适应bpemb词向量
    将来会加入英语的分词包。
    """

    def __init__(self, text: str = None, language: str = None,
                 labels: Union[List[Label], List[str]] = None,
                 bp_encoder: BPEmb = None):

        super(Sentence, self).__init__()

        self.tokens: List[Token] = []
        self.labels: List[Label] = []
        if labels is not None: self.add_labels(labels)

        self._embeddings: Dict = {}
        self.lang = language

        if text is not None:
            if language in ['UY','BP']:
                if language == 'BP':
                    text = ' '.join(bp_encoder.encode(text))
                if language == 'UY':
                    text = uy_preprocess(text)
                word = ''
                for index, char in enumerate(text):
                    if char == ' ':
                        if len(word) > 0:
                            token = Token(word, start_position=index - len(word),lang = language)
                            self.add_token(token)

                        word = ''
                    else:
                        word += char
                index += 1
                if len(word) > 0:
                    token = Token(word, start_position=index - len(word),lang = language)
                    self.add_token(token)

            elif language == 'EN':

                tokens = []
                sentences = split_single(text)
                for sentence in sentences:
                    contractions = split_contractions(word_tokenizer(sentence))
                    tokens.extend(contractions)

                index = text.index
                running_offset = 0
                last_word_offset = -1
                last_token = None
                for word in tokens:
                    try:
                        word_offset = index(word, running_offset)
                        start_position = word_offset
                    except:
                        word_offset = last_word_offset + 1
                        start_position = running_offset + 1 if running_offset > 0 else running_offset

                    token = Token(word, start_position=start_position)
                    self.add_token(token)

                    if word_offset - 1 == last_word_offset and last_token is not None:
                        last_token.whitespace_after = False

                    word_len = len(word)
                    running_offset = word_offset + word_len
                    last_word_offset = running_offset - 1
                    last_token = token

            elif language == 'CN_char':
                for index, char in enumerate(text):
                    token = Token(char, start_position=index)
                    self.add_token(token)

            elif language == 'CN_token':
                seg_list = list(jieba.tokenize(text))
                for t in seg_list:
                    token = Token(t[0], start_position=t[1])
                    self.add_token(token)

            elif language == 'PY':
                for index, char in enumerate(text):
                    token = Token(char, start_position=index,lang=language)
                    self.add_token(token)

    def get_token(self, token_id: int) -> Token:
        for token in self.tokens:
            if token.idx == token_id:
                return token

    def add_token(self, token: Token):
        self.tokens.append(token)

        token.sentence = self
        if token.idx is None:
            token.idx = len(self.tokens)

    def get_spans(self, tag_type: str, min_score=-1) -> List[Span]:

        spans: List[Span] = []

        current_span = []

        tags = defaultdict(lambda: 0.0)

        previous_tag_value: str = 'O'
        for token in self:

            tag: Label = token.get_tag(tag_type)
            tag_value = tag.value

            if tag_value == '' or tag_value == 'O':
                tag_value = 'O-'

            if tag_value[0:2] not in ['B-', 'I-', 'O-', 'E-', 'S-']:
                tag_value = 'S-' + tag_value

            in_span = False
            if tag_value[0:2] not in ['O-']:
                in_span = True

            starts_new_span = False
            if tag_value[0:2] in ['B-', 'S-']:
                starts_new_span = True

            if previous_tag_value[0:2] in ['S-'] and previous_tag_value[2:] != tag_value[2:] and in_span:
                starts_new_span = True

            if (starts_new_span or not in_span) and len(current_span) > 0:
                scores = [t.get_tag(tag_type).score for t in current_span]
                span_score = sum(scores) / len(scores)
                if span_score > min_score:
                    spans.append(Span(
                        current_span,
                        tag=sorted(tags.items(), key=lambda k_v: k_v[1], reverse=True)[0][0],
                        score=span_score)
                    )
                current_span = []
                tags = defaultdict(lambda: 0.0)

            if in_span:
                current_span.append(token)
                weight = 1.1 if starts_new_span else 1.0
                tags[tag_value[2:]] += weight

            previous_tag_value = tag_value

        if len(current_span) > 0:
            scores = [t.get_tag(tag_type).score for t in current_span]
            span_score = sum(scores) / len(scores)
            if span_score > min_score:
                spans.append(Span(
                    current_span,
                    tag=sorted(tags.items(), key=lambda k_v: k_v[1], reverse=True)[0][0],
                    score=span_score)
                )

        return spans

    def add_label(self, label: Union[Label, str]):
        if type(label) is Label:
            self.labels.append(label)

        elif type(label) is str:
            self.labels.append(Label(label))

    def add_labels(self, labels: Union[List[Label], List[str]]):
        for label in labels:
            self.add_label(label)

    def get_label_names(self) -> List[str]:
        return [label.value for label in self.labels]

    @property
    def embedding(self):
        return self.get_embedding()

    def set_embedding(self, name: str, vector):
        self._embeddings[name] = vector.cpu()

    def get_embedding(self) -> torch.autograd.Variable:
        embeddings = []
        for embed in sorted(self._embeddings.keys()):
            embedding = self._embeddings[embed]
            embeddings.append(embedding)

        if embeddings:
            return torch.cat(embeddings, dim=0)

        return torch.FloatTensor()

    def clear_embeddings(self, also_clear_word_embeddings: bool = True):
        self._embeddings: Dict = {}

        if also_clear_word_embeddings:
            for token in self:
                token.clear_embeddings()

    def cpu_embeddings(self):
        for name, vector in self._embeddings.items():
            self._embeddings[name] = vector.cpu()

    def to_tagged_string(self, main_tag=None) -> str:
        list = []
        for token in self.tokens:
            list.append(token.text)

            tags: List[str] = []
            for tag_type in token.tags.keys():

                if main_tag is not None and main_tag != tag_type: continue

                if token.get_tag(tag_type).value == '' or token.get_tag(tag_type).value == 'O': continue
                tags.append(token.get_tag(tag_type).value)
            all_tags = '<' + '/'.join(tags) + '>'
            if all_tags != '<>':
                list.append(all_tags)
        return ' '.join(list)

    def to_tokenized_string(self, lang: str = None) -> str:
        if lang == 'UY':
            return ' '.join([t.latin for t in self.tokens])
        elif lang == 'PY':
            return ' '.join([t.pinyin for t in self.tokens])
        else:
            return ' '.join([t.text for t in self.tokens])

    def to_plain_string(self, lang: str = None):
        plain = ''
        for token in self.tokens:
            if lang == 'UY':
                plain += token.latin
            elif lang == 'PY':
                plain += token.pinyin
            else:
                plain += token.text
            if token.whitespace_after:
                plain += ' '
        return plain.rstrip()

    def convert_tag_scheme(self, tag_type: str = 'ner', target_scheme: str = 'iob'):

        tags: List[Label] = []
        for token in self.tokens:
            token: Token = token
            tags.append(token.get_tag(tag_type))

        if target_scheme == 'iob':
            iob2(tags)

        if target_scheme == 'iobes':
            iob2(tags)
            tags = iob_iobes(tags)

        for index, tag in enumerate(tags):
            self.tokens[index].add_tag(tag_type, tag.value)

    def infer_space_after(self):

        last_token = None
        quote_count: int = 0

        for token in self.tokens:
            if token.text == '"':
                quote_count += 1
                if quote_count % 2 != 0:
                    token.whitespace_after = False
                elif last_token is not None:
                    last_token.whitespace_after = False

            if last_token is not None:

                if token.text in ['.', ':', ',', ';', ')', 'n\'t', '!', '?', '،', '؛', '؟']:
                    last_token.whitespace_after = False

                if token.text.startswith('\''):
                    last_token.whitespace_after = False

            if token.text in ['(']:
                token.whitespace_after = False

            last_token = token
        return self

    def to_original_text(self) -> str:
        str = ''
        pos = 0
        for t in self.tokens:
            while t.start_pos != pos:
                str += ' '
                pos += 1

            str += t.text
            pos += len(t.text)

        return str

    def to_dict(self, tag_type: str = None):
        labels = []
        entities = []

        if tag_type:
            entities = [span.to_dict() for span in self.get_spans(tag_type)]
        if self.labels:
            labels = [l.to_dict() for l in self.labels]

        return {
            'text': self.to_original_text(),
            'labels': labels,
            'entities': entities
        }

    def __getitem__(self, idx: int) -> Token:
        return self.tokens[idx]

    def __iter__(self):
        return iter(self.tokens)

    def __repr__(self):
        return 'Sentence: "{}" - {} Tokens'.format(' '.join([t.text for t in self.tokens]), len(self))

    def __copy__(self):
        s = Sentence()
        for token in self.tokens:
            nt = Token(token.text)
            for tag_type in token.tags:
                nt.add_tag(tag_type, token.get_tag(tag_type).value, token.get_tag(tag_type).score)

            s.add_token(nt)
        return s

    def __str__(self) -> str:
        return 'Sentence: "{}" - {} Tokens'.format(' '.join([t.text for t in self.tokens]), len(self))

    def __len__(self) -> int:
        return len(self.tokens)


class Corpus:
    """
    数据集类
    """
    @property
    @abstractmethod
    def train(self) -> List[Sentence]:
        pass

    @property
    @abstractmethod
    def dev(self) -> List[Sentence]:
        pass

    @property
    @abstractmethod
    def test(self) -> List[Sentence]:
        pass

    @abstractmethod
    def downsample(self, percentage: float = 0.1, only_downsample_train=False):
        pass

    @abstractmethod
    def get_all_sentences(self) -> List[Sentence]:
        pass

    @abstractmethod
    def make_tag_dictionary(self, tag_type: str) -> Dictionary:
        pass

    @abstractmethod
    def make_label_dictionary(self) -> Dictionary:
        pass


class TaggedCorpus(Corpus):
    """
    标注了的数据集类
    """
    def __init__(self, train: List[Sentence], dev: List[Sentence], test: List[Sentence], name: str = 'corpus'):
        self._train: List[Sentence] = train
        self._dev: List[Sentence] = dev
        self._test: List[Sentence] = test
        self.name: str = name

    @property
    def train(self) -> List[Sentence]:
        return self._train

    @property
    def dev(self) -> List[Sentence]:
        return self._dev

    @property
    def test(self) -> List[Sentence]:
        return self._test

    def downsample(self, percentage: float = 0.1, only_downsample_train=False):

        self._train = self._downsample_to_proportion(self.train, percentage)
        if not only_downsample_train:
            self._dev = self._downsample_to_proportion(self.dev, percentage)
            self._test = self._downsample_to_proportion(self.test, percentage)

        return self

    def get_all_sentences(self) -> List[Sentence]:
        all_sentences: List[Sentence] = []
        all_sentences.extend(self.train)
        all_sentences.extend(self.dev)
        all_sentences.extend(self.test)
        return all_sentences

    def convert_scheme(self, type:str='ner',scheme:str='iob'):
        for sent in self.train:
            sent.convert_tag_scheme(type, scheme)
        for sent in self.test:
            sent.convert_tag_scheme(type, scheme)
        for sent in self.dev:
            sent.convert_tag_scheme(type, scheme)
        return True

    def make_tag_dictionary(self, tag_type: str) -> Dictionary:

        # Make the tag dictionary
        tag_dictionary: Dictionary = Dictionary()
        tag_dictionary.add_item('O')
        for sentence in self.get_all_sentences():
            for token in sentence.tokens:
                token: Token = token
                tag_dictionary.add_item(token.get_tag(tag_type).value)
        tag_dictionary.add_item('<START>')
        tag_dictionary.add_item('<STOP>')
        return tag_dictionary

    def make_label_dictionary(self) -> Dictionary:

        labels = set(self._get_all_label_names())

        label_dictionary: Dictionary = Dictionary(add_unk=False)
        for label in labels:
            label_dictionary.add_item(label)

        return label_dictionary

    def make_vocab_dictionary(self, max_tokens=-1, min_freq=1) -> Dictionary:

        tokens = self._get_most_common_tokens(max_tokens, min_freq)

        vocab_dictionary: Dictionary = Dictionary()
        for token in tokens:
            vocab_dictionary.add_item(token)

        return vocab_dictionary

    def _get_most_common_tokens(self, max_tokens, min_freq) -> List[str]:
        tokens_and_frequencies = Counter(self._get_all_tokens())
        tokens_and_frequencies = tokens_and_frequencies.most_common()

        tokens = []
        for token, freq in tokens_and_frequencies:
            if (min_freq != -1 and freq < min_freq) or (max_tokens != -1 and len(tokens) == max_tokens):
                break
            tokens.append(token)
        return tokens

    def _get_all_label_names(self) -> List[str]:
        return [label.value for sent in self.train for label in sent.labels]

    def _get_all_tokens(self) -> List[str]:
        tokens = list(map((lambda s: s.tokens), self.train))
        tokens = [token for sublist in tokens for token in sublist]
        return list(map((lambda t: t.text), tokens))

    def _downsample_to_proportion(self, list: List, proportion: float):

        counter = 0.0
        last_counter = None
        downsampled: List = []

        for item in list:
            counter += proportion
            if int(counter) != last_counter:
                downsampled.append(item)
                last_counter = int(counter)
        return downsampled

    def obtain_statistics(self, tag_type: str = None) -> dict:

        return {
            "TRAIN": self._obtain_statistics_for(self.train, "TRAIN", tag_type),
            "TEST": self._obtain_statistics_for(self.test, "TEST", tag_type),
            "DEV": self._obtain_statistics_for(self.dev, "DEV", tag_type),
        }

    @staticmethod
    def _obtain_statistics_for(sentences, name, tag_type) -> dict:
        if len(sentences) == 0:
            return {}

        classes_to_count = TaggedCorpus._get_class_to_count(sentences)
        tags_to_count = TaggedCorpus._get_tag_to_count(sentences, tag_type)
        tokens_per_sentence = TaggedCorpus._get_tokens_per_sentence(sentences)

        label_size_dict = {}
        for l, c in classes_to_count.items():
            label_size_dict[l] = c

        tag_size_dict = {}
        for l, c in tags_to_count.items():
            tag_size_dict[l] = c

        return {
            'dataset': name,
            'total_number_of_documents': len(sentences),
            'number_of_documents_per_class': label_size_dict,
            'number_of_tokens_per_tag': tag_size_dict,
            'number_of_tokens': {
                'total': sum(tokens_per_sentence),
                'min': min(tokens_per_sentence),
                'max': max(tokens_per_sentence),
                'avg': sum(tokens_per_sentence) / len(sentences)
            }
        }

    @staticmethod
    def _get_tokens_per_sentence(sentences):
        return list(map(lambda x: len(x.tokens), sentences))

    @staticmethod
    def _get_class_to_count(sentences):
        class_to_count = defaultdict(lambda: 0)
        for sent in sentences:
            for label in sent.labels:
                class_to_count[label.value] += 1
        return class_to_count

    @staticmethod
    def _get_tag_to_count(sentences, tag_type):
        tag_to_count = defaultdict(lambda: 0)
        for sent in sentences:
            for word in sent.tokens:
                if tag_type in word.tags:
                    label = word.tags[tag_type]
                    tag_to_count[label.value] += 1
        return tag_to_count

    def __str__(self) -> str:
        return 'TaggedCorpus: %d train + %d dev + %d test sentences' % (len(self.train), len(self.dev), len(self.test))


def iob2(tags):

    for i, tag in enumerate(tags):
        if tag.value == 'O':
            continue
        split = tag.value.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1].value == 'O':
            tags[i].value = 'B' + tag.value[1:]
        elif tags[i - 1].value[1:] == tag.value[1:]:
            continue
        else:
            tags[i].value = 'B' + tag.value[1:]
    return True

def uy_preprocess(text):

    text = re.sub('،' ,' ، ',text)
    text = re.sub('\.', ' . ', text)
    text = re.sub('!', ' ! ', text)
    text = re.sub('؟', ' ؟ ', text)
    text = re.sub('\?', ' ? ', text)
    text = re.sub('\(' ,'( ',text)
    text = re.sub('\)' ,' )',text)
    text = re.sub('»' ,' »',text)
    text = re.sub('«' ,'« ',text)
    text = re.sub(':' ,' :',text)
    text = re.sub('"' ,' " ',text)
    text = re.sub('><' ,'> <',text)
    text = re.sub(r'( )*-( )*', '-', text)

    return text

def iob_iobes(tags):

    new_tags = []
    for i, tag in enumerate(tags):
        if tag.value == 'O':
            new_tags.append(tag.value)
        elif tag.value.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
                    tags[i + 1].value.split('-')[0] == 'I':
                new_tags.append(tag.value)
            else:
                new_tags.append(tag.value.replace('B-', 'S-'))
        elif tag.value.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].value.split('-')[0] == 'I':
                new_tags.append(tag.value)
            else:
                new_tags.append(tag.value.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


class MultiCorpus(Corpus):
    """
    多数据集类
    """
    def __init__(self, corpora: List[TaggedCorpus]):
        self.corpora: List[TaggedCorpus] = corpora

    @property
    def train(self) -> List[Sentence]:
        train: List[Sentence] = []
        for corpus in self.corpora:
            train.extend(corpus.train)
        return train

    @property
    def dev(self) -> List[Sentence]:
        dev: List[Sentence] = []
        for corpus in self.corpora:
            dev.extend(corpus.dev)
        return dev

    @property
    def test(self) -> List[Sentence]:
        test: List[Sentence] = []
        for corpus in self.corpora:
            test.extend(corpus.test)
        return test

    def __str__(self):
        return '\n'.join([str(corpus) for corpus in self.corpora])

    def get_all_sentences(self) -> List[Sentence]:
        sentences = []
        for corpus in self.corpora:
            sentences.extend(corpus.get_all_sentences())
        return sentences

    def downsample(self, percentage: float = 0.1, only_downsample_train=False):

        for corpus in self.corpora:
            corpus.downsample(percentage, only_downsample_train)

        return self

    def make_tag_dictionary(self, tag_type: str) -> Dictionary:

        # Make the tag dictionary
        tag_dictionary: Dictionary = Dictionary()
        tag_dictionary.add_item('O')
        for corpus in self.corpora:
            for sentence in corpus.get_all_sentences():
                for token in sentence.tokens:
                    token: Token = token
                    tag_dictionary.add_item(token.get_tag(tag_type).value)
        tag_dictionary.add_item('<START>')
        tag_dictionary.add_item('<STOP>')
        return tag_dictionary

    def make_label_dictionary(self) -> Dictionary:

        label_dictionary: Dictionary = Dictionary(add_unk=False)
        for corpus in self.corpora:
            labels = set(corpus._get_all_label_names())

            for label in labels:
                label_dictionary.add_item(label)

        return label_dictionary
