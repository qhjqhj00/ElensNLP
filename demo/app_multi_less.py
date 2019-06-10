from flask import Flask, request, Response
from lensnlp.utilis.data_preprocess import cn_prepare
from lensnlp.models import SequenceTagger
import json
import requests
import threading
import copy


app = Flask(__name__)


tagger_t3 = SequenceTagger.load('cn_s')
tagger_t5 = SequenceTagger.load('cn_5')
"""with open('enter.json','r') as f:
    enter_db = json.load(f)"""

print('中文模型已加载...')


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
        map = {'PER': '人名', 'LOC': '地名', 'ORG': '机构名','CAT': '品类', 'TITLE': '职位'}
        result = []
        for s in self.sentences:
            tem = s.to_dict(tag_type='ner')
            for m in range(len(tem['entities'])):
                """if tem['entities'][m]['type'] == 'MOV':
                    if tem['entities'][m]['text'] in enter_db:
                        tem['entities'][m]['type'] = enter_db[tem['entities'][m]['text']]
                    else:
                        tem['entities'][m]['type'] = map[tem['entities'][m]['type']]
                else:"""
                if tem['entities'][m]['type'] in map:
                    tem['entities'][m]['type'] = map[tem['entities'][m]['type']]
            result.append(tem)
        return result


def get_more(text):
    tag = {'cri':'罪名','med':'药名','ill':'疾病名','idi':'成语'}
    r = requests.post('http://192.168.1.148:9003/page/segment', data=text.encode('utf-8')).json()
    r = [ent for ent in r if ent['type'] in tag]
    for ent in r:
        ent['type'] = tag[ent['type']]
        ent['start_pos'] = int(ent['start_pos'])
        ent['end_pos'] = int(ent['end_pos'])
    return r


@app.route('/cn/', methods=['POST', 'GET'])
def cn_ner():
    input_json = request.get_json(force=True)
    data = input_json['cn_data']
    print('GET MESSAGES:', data)
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
        s = json.dumps({'cn_data':sent['text']})
        r = requests.get('http://192.168.1.147:4088/bra/', data=s)
        result[i]['entities'].extend(r.json()[0])
        result[i]['entities'].extend(get_more(sent['text']))

    return Response(json.dumps(result, ensure_ascii=False), mimetype='application/json; charset=utf-8')


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=1111,threaded=True)
