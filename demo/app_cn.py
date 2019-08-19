from flask import Flask, request, Response
from lensnlp.utilis.data_preprocess import cn_prepare,en_prepare, clf_preprocess,uy_prepare
from lensnlp.models import SequenceTagger, TextClassifier
import json
import sys
import os
from pathlib import Path

tagger = SequenceTagger.load('cn_s')
app = Flask(__name__)


@app.route('/cn/', methods=['POST', 'GET'])
def model():
    input_json = request.get_json(force=True)
    data = input_json['cn_data']

    sentences = cn_prepare(data)
    tagger.predict(sentences)

    result = []
    for s in sentences:

        neural_result = s.to_dict(tag_type='ner')
        result.append(neural_result)

    return Response(json.dumps(result, ensure_ascii=False), mimetype='application/json; charset=utf-8')


@app.route('/cn_ner/',methods=['GET'])
def model_1():
    data = request.args.get("cn_data")
    sentences = cn_prepare(data)
    tagger.predict(sentences)

    result = []
    for s in sentences:

        neural_result = s.to_dict(tag_type='ner')
        result.append(neural_result)

    return Response(json.dumps(result, ensure_ascii=False), mimetype='application/json; charset=utf-8')

if __name__ == '__main__':
    app.run(host ='0.0.0.0',port=4500,threaded=True)

