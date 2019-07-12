from lensnlp.models import SequenceTagger, TextClassifier
from lensnlp.utils.data_preprocess import cn_prepare, en_prepare, uy_prepare, clf_preprocess


class ner:
    """实体识别模型，提供中文，英文，维吾尔语三个语种的预训练模型。

        Parameters
        ----------

        language : str
            选择语种

        Examples
        --------
         >>> from lensnlp import ner
         >>> sent = '北京一览群智。'
         >>> cn_tagger = ner('cn')
         >>> cn_tagger.predict(sent)

        """

    def __init__(self,language):

        self.language = language
        if self.language == 'cn':
            self.tagger = SequenceTagger.load('cn_s')
        elif self.language == 'en':
            self.tagger = SequenceTagger.load('en_s')
        elif self.language == 'uy':
            self.tagger = SequenceTagger.load('uy_s')
        else:
            raise ValueError('Not yet!')

    def preprocess(self,text):
        if self.language == 'cn':
            return cn_prepare(text)
        elif self.language == 'en':
            return en_prepare(text)
        elif self.language == 'uy':
            return uy_prepare(text)
        else:
            raise ValueError('Not yet!')

    def predict(self,text):
        sentences = self.preprocess(text)
        self.tagger.predict(sentences)
        result = []
        for s in sentences:
            neural_result = s.to_dict(tag_type='ner')
            result.append(neural_result)
        return result


class clf:
    """文本分类模型，情感分析，提供中文，英文，维吾尔语三个语种的预训练模型。

        Parameters
        ----------

        language : str
            选择语种

        Examples
        --------
         >>> from lensnlp import clf
         >>> sent = '北京一览群智。'
         >>> cn_clf = clf('cn')
         >>> cn_clf.predict(sent)

        """


    def __init__(self,language):
        self.language = language
        if self.language == 'cn_clf':
            self.clf = TextClassifier.load('cn_clf')
        elif self.language == 'en_clf':
            self.clf = TextClassifier.load('en_clf')
        elif self.language == 'uy_clf':
            self.clf = TextClassifier.load('uy_clf')
        elif self.language == 'cn_emo':
            self.clf = TextClassifier.load('cn_emo')
        elif self.language == 'en_emo':
            self.clf = TextClassifier.load('en_emo')
        elif self.language == 'uy_emo':
            self.clf = TextClassifier.load('uy_emo')

        else:
            raise ValueError('Not yet!')

    def preprocess(self,text):
        return clf_preprocess(text,self.language)

    def predict(self,text):
        sentences = self.preprocess(text)
        self.clf.predict(sentences)
        result = []
        for s in sentences:
            if self.language == 'cn_clf':
                result.append(s.labels[0].value)
            elif self.language == 'en_clf':
                result.append(int(s.labels[0].value))
            elif self.language == 'uy_clf':
                result.append(int(s.labels[0].value))
            elif 'emo' in self.language:
                result.append(int(s.labels[0].value))
            else:
                raise ValueError('Not yet!')

        return result
