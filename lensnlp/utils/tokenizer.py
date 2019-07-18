from typing import List
import langid
from lensnlp.utils.data import Token
import jieba
from segtok.segmenter import split_single
from segtok.tokenizer import split_contractions
from segtok.tokenizer import word_tokenizer


class Tokenizer:
    def __init__(self, language_type: str = None, example: str = None, sp_op: str = None):

        if language_type is None and example is None:
            raise ValueError("Must specify language type or provides an example")

        if sp_op is not None and sp_op not in ['char']:
            raise ValueError("Not support the operation yet")

        if language_type is not None:
            self.language_type = language_type
        else:
            self.language_type = langid.classify(example)[0]

        self.sp_op = sp_op

    def word_tokenizer(self, text) -> List[Token]:
        tokenized = []
        if self.language_type == 'zh':
            if self.sp_op == 'char':
                for index, char in enumerate(text):
                    token = Token(char, start_position=index)
                    tokenized.append(token)
            else:
                seg_list = list(jieba.tokenize(text))
                for t in seg_list:
                    token = Token(t[0], start_position=t[1])
                    tokenized.append(token)

        elif self.language_type == 'ug':
            text = uy_preprocess(text)
            word = ''
            for index, char in enumerate(text):
                if char == ' ':
                    if len(word) > 0:
                        token = Token(word, start_position=index - len(word), lang=language)
                        self.add_token(token)

                    word = ''
                else:
                    word += char
            index += 1
            if len(word) > 0:
                token = Token(word, start_position=index - len(word), lang=language)
                tokenized.append(token)

        elif self.language_type == 'en':
            tokenized = []
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
                tokenized.append(token)

                if word_offset - 1 == last_word_offset and last_token is not None:
                    last_token.whitespace_after = False

                word_len = len(word)
                running_offset = word_offset + word_len
                last_word_offset = running_offset - 1
                last_token = token

        return tokenized

    def sentence_split(self, text) -> List[str]:
        pass

    @staticmethod
    def uy_preprocess(text):
        """维吾尔语预处理"""
        text = re.sub('،', ' ، ', text)
        text = re.sub(r'\.', ' . ', text)
        text = re.sub('!', ' ! ', text)
        text = re.sub('؟', ' ؟ ', text)
        text = re.sub(r'\?', ' ? ', text)
        text = re.sub(r'\(', '( ', text)
        text = re.sub(r'\)', ' )', text)
        text = re.sub('»', ' »', text)
        text = re.sub('«', '« ', text)
        text = re.sub(':', ' :', text)
        text = re.sub('"', ' " ', text)
        text = re.sub('><', '> <', text)
        text = re.sub(r'( )*-( )*', '-', text)

        return text
