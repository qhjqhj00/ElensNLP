from lensnlp.models import SequenceTagger, TextClassifier
from lensnlp.utilis.data_preprocess import cn_prepare, en_prepare, uy_prepare, clf_preprocess
from lensnlp.hyper_parameters import id_to_label_15


class ner:
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
    def __init__(self,language):
        self.language = language
        if self.language == 'cn':
            self.clf = TextClassifier.load('cn_15')
        else:
            raise ValueError('Not yet!')

    def preprocess(self,text):
        if self.language == 'cn':
            return clf_preprocess(text)
        else:
            raise ValueError('Not yet!')

    def predict(self,text):
        sentences = self.preprocess(text)
        self.clf.predict(sentences)
        result = []
        for s in sentences:
            result.append(id_to_label_15[int(s.labels[0].value)])
        return result
