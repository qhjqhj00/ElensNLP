from flask import Flask, request, Response
from lensnlp.utilis.data_preprocess import clf_preprocess
from lensnlp.models import TextClassifier
from lensnlp.hyper_parameters import cn_clf
import json
import os

CACHE_ROOT = os.path.expanduser(os.path.join('~', '.lensnlp'))

app = Flask(__name__)

clf = TextClassifier.load('cn_clf')


@app.route('/lensnlp/', methods=['POST', 'GET'])
def model():
    input_json = request.get_json(force=True)
    data = input_json['data']
    sentences = clf_preprocess(data,'cn')
    clf.predict(sentences)

    result = [cn_clf[int(s.labels[0].value)] for s in sentences]

    return Response(json.dumps(result, ensure_ascii=False), mimetype='application/json; charset=utf-8')


if __name__ == '__main__':
    app.run(host ='0.0.0.0',port=1233,threaded=True)
