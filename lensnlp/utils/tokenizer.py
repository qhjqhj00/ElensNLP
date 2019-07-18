from typing import List
import langid


class Tokenizer:
    def __init__(self, sp_op: str = None):
        self.sp_op = sp_op

    def word_tokenizer(self, text) -> List[str]:
        pass

    def sentence_split(self, text) -> List[str]:
        pass

    if text is not None:
        if language == 'UY':
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
                token = Token(char, start_position=index, lang=language)
                self.add_token(token)