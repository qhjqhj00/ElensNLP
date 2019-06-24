from flask import Flask, request, Response
from lensnlp.utilis.data_preprocess import cn_prepare
from lensnlp.models import SequenceTagger
import json
import requests
import threading
import copy
import re


app = Flask(__name__)


tagger_t3 = SequenceTagger.load('cn_s')
tagger_t5 = SequenceTagger.load('cn_5')


def regx_ner(text):
    regex_dict = {'MAIL':'[0-9a-zA-Z_]{0,19}@[0-9a-zA-Z\.]{1,18}\.[a-z]{1,6}',
              'TELE':'\(?0\d{2,3}[)-]?\d{7,8}|(?:(\+\d{2}))?(\d{1,3})\d{8}',
              'WEB':'([a-zA-Z]+://)?([a-zA-Z0-9._-]{1,66})\.[a-z]{1,5}(:[0-9]{1,4})*(/[a-zA-Z0-9&%_./-~-]*)*',
              'TAG':'#([^#]*){1,32}#'}
    text=text.replace('．','.')
    text=text.replace('－','-')
    result = []
    for key in regex_dict:
        res = re.finditer(regex_dict[key],text)
        tmp_result = [{'text':t.group(),'start_pos':t.start(),
                   'end_pos':t.end(),'type':key} for t in res]
        if key == 'WEB':
            for ent in tmp_result:
                if ent['start_pos'] != 0:
                    if text[ent['start_pos']-1] != '@':
                        result.append(ent)
                else:
                    result.append(ent)
    return result

class decode_thread(threading.Thread):

    def __init__(self, type, sentences):
        threading.Thread.__init__(self)
        self.type = type
        self.sentences = copy.deepcopy(sentences)

    def run(self):
        if self.type == 'cn_s':
            tagger_t3.predict(self.sentences)
        if self.type == 'cn_5':
            tagger_t5.predict(self.sentences)

    def get_result(self):
        map = {'PER': 'PER', 'LOC': 'LOC', 'ORG': 'ORG','CAT': 'CAT', 'TITLE': 'TITLE'}
        result = []
        for s in self.sentences:
            tem = s.to_dict(tag_type='ner')
            tem['entities'] = [ent for ent in tem['entities'] if ent['type'] in map]
            for m in range(len(tem['entities'])):
                tem['entities'][m]['type'] = map[tem['entities'][m]['type']]
            result.append(tem)
        return result


def get_more(text):
    tag = {'cri':'CRIME','med':'MED','ill':'ILL','idi':'IDIOM'}
    r = requests.post('http://192.168.1.147:9003/page/segment', data=text.encode('utf-8')).json()
    r = [ent for ent in r if ent['type'] in tag]
    for ent in r:
        ent['type'] = tag[ent['type']]
        ent['text'] = ent.pop('entity')
        ent['start_pos'] = int(ent['start_pos'])
        ent['end_pos'] = int(ent['end_pos'])
    return r


@app.route('/cn/', methods=['POST', 'GET'])
def cn_ner():
    input_json = request.get_json(force=True)
    data = input_json['doc']
    sentences = cn_prepare(data)

    models = ['cn_s', 'cn_5']
    names = locals()
    threads = []

    for model_tag in models:
        names['tagger%s' % model_tag] = decode_thread(model_tag, sentences)
        threads.append(names['tagger%s' % model_tag])

    result = []

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for thread in threads:
        tem_result = thread.get_result()
        if len(result) > 0:
            for i,sent in enumerate(tem_result):
                result[i]['entities'].extend(sent['entities'])
        else:
            result = tem_result

    for i,sent in enumerate(result):
        if len(sent['text']) >0:
            result[i]['entities'].extend(get_more(sent['text']))
            result[i]['entities'].extend(regx_ner(sent['text']))

    return Response(json.dumps(result, ensure_ascii=False), mimetype='application/json; charset=utf-8')


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=1111,threaded=True)

