from flask import Flask, request, Response
from lensnlp.utilis.data_preprocess import cn_prepare,en_prepare, clf_preprocess,uy_prepare
from lensnlp.models import SequenceTagger, TextClassifier
from lensnlp.hyper_parameters import cn_clf
import json
import sys
import os
from pathlib import Path

CACHE_ROOT = os.path.expanduser(os.path.join('~', '.lensnlp'))

if len(sys.argv) >= 2:
    config_path = sys.argv[1]
    with open(config_path,'r') as f:
        config = json.load(f)['predict']
    model_dict = {}
    for model_name in config:
        names = locals()
        if config[model_name]['execute'] is True:
            language = config[model_name]['language']
            type = config[model_name]['type']
            model_path = config[model_name]['model_path']
            if type == 'ner':
                names['tagger_%s'%model_name] = SequenceTagger.load(Path(CACHE_ROOT) / model_path)
                model_dict[model_name] = names['tagger_%s'%model_name]
            if type == 'clf':
                names['clf_%s' % model_name] = TextClassifier.load(Path(CACHE_ROOT) / model_path)
                model_dict[model_name] = names['clf_%s'%model_name]
else:
    raise ValueError('Please Specify Configure File!')

app = Flask(__name__)


@app.route('/lensnlp/', methods=['POST', 'GET'])
def model():
    input_json = request.get_json(force=True)
    data = input_json['data']
    model_name = input_json['name']
    language = config[model_name]['language']
    task_type = config[model_name]['type']

    # print('GET MESSAGES:', data)
    if task_type == 'ner':
        if language == 'cn':
            sentences = cn_prepare(data)
        elif language == 'en':
            sentences = en_prepare(data)
        elif language == 'uy':
            sentences = uy_prepare(data)
        else:
            raise ValueError('Not Yet!')
    elif task_type == 'clf':
        sentences = clf_preprocess(data,'cn')
    else:
        raise NameError('Not Yet!')

    model_dict[model_name].predict(sentences)

    result = []
    for s in sentences:
        if task_type == 'ner':
            neural_result = s.to_dict(tag_type='ner')
            result.append(neural_result)
        elif task_type == 'clf':
            result.append(cn_clf[int(s.labels[0].value)])
        else:
            raise NameError('Not Yet!')
    return Response(json.dumps(result, ensure_ascii=False), mimetype='application/json; charset=utf-8')


if len(sys.argv) == 3:
    set_port = sys.argv[2]
else:
    set_port = 4000

if __name__ == '__main__':
    app.run(host ='0.0.0.0',port=set_port,threaded=True)

