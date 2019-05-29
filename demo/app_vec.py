from gensim.models import KeyedVectors
from flask import Flask, request, Response, make_response

import json
import os


CACHE_ROOT = os.path.expanduser(os.path.join('~', '.lensnlp'))

word_embeddings = KeyedVectors.load(CACHE_ROOT+'/language_model/cn_glove_300d')

app = Flask(__name__)


@app.route('/vec/', methods=['POST', 'GET'])
def similarity():
    input_json = request.get_json(force=True)

    src = input_json['src']
    if src in word_embeddings:
        words_list = word_embeddings.similar_by_word(src)
        result = {}

        for w,s in words_list:
            result[w]=str(s)
    else:
        result = 'fail'
    
    return Response(json.dumps(result, ensure_ascii=False), mimetype='application/json; charset=utf-8')

if __name__ == '__main__':
    app.run(host ='0.0.0.0',port=10100,threaded=True)
