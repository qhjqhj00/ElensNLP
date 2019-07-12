from flask import Flask, request, Response, make_response
from lensnlp.utils.data_preprocess import clf_preprocess
from lensnlp.models import TextClassifier
import json
import os

CACHE_ROOT = os.path.expanduser(os.path.join('~', '.lensnlp'))

app = Flask(__name__)


clf = TextClassifier.load('cn_clf')
emo = TextClassifier.load('cn_emo')


@app.route('/cn_clf/', methods=['POST', 'GET'])
def clf_1():
    input_json = request.get_json(force=True)
    data = input_json['cn_data']
    sentences = clf_preprocess(data,'CN_token')
    clf.predict(sentences)

    result = [s.labels[0].value for s in sentences]

    return Response(json.dumps(result, ensure_ascii=False), mimetype='application/json; charset=utf-8')


@app.route('/cn_emo/', methods=['POST', 'GET'])
def emo_1():
    input_json = request.get_json(force=True)
    data = input_json['cn_data']
    sentences = clf_preprocess(data,'CN_token')
    emo.predict(sentences)

    result = [s.labels[0].value for s in sentences]

    return Response(json.dumps(result, ensure_ascii=False), mimetype='application/json; charset=utf-8')


@app.route('/clf_cn/',methods=['GET'])
def clf_2():
    data = request.args.get("cn_data")
    sentences = clf_preprocess(data,'CN_token')
    clf.predict(sentences)

    result = [s.labels[0].value for s in sentences]

    return Response(json.dumps(result, ensure_ascii=False), mimetype='application/json; charset=utf-8')


@app.route('/emo_cn/',methods=['GET'])
def emo_2():
    data = request.args.get("cn_data")
    sentences = clf_preprocess(data,'CN_token')
    emo.predict(sentences)

    result = [s.labels[0].value for s in sentences]

    return Response(json.dumps(result, ensure_ascii=False), mimetype='application/json; charset=utf-8')


if __name__ == '__main__':
    app.run(host ='0.0.0.0',port=4888,threaded=True)

