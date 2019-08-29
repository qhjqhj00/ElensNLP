import pymongo
import requests
import re
from flask import Flask, request, Response
from lensnlp.utils.data_preprocess import uy_prepare,uy_segmentation
from lensnlp.models import SequenceTagger
import json
import os


def trans(text):
    #sor_lan源语言
    #目标语言
    #text翻译文档
    print('小牛')
    source = 'uy'
    target = 'zh'
    url = 'http://demo.niutrans.vip:5020/NiuTransServer/test?&from=%s&to=%s&src_text=%s'%(source,target,text)#调用小牛接口
    content = requests.get(url)#post访问
    content.encoding = 'utf-8'#utf-8编码
    content.json()
    return content.json()['tgt_text'].strip()


def translator(src):
    myclient = pymongo.MongoClient("192.168.1.147",port=27017,username='admin', password='password')
    mydb = myclient["uyghur"]
    myset = mydb.dict
    result = list(myset.find({"ug":src}))
    print(result)
    if len(result) != 0:
        return result[0]['zh']
    else:
        trs = trans(src)
        u = dict(ug = src,zh=trs)
        myset.insert(u)
        return trs


def find_time(text):
    a = ['[0-9]{1,2}-ئاينىڭ',
         '[0-9]{1,2}-كۈنى',
         '[0-9]{2,4}-يىلى',
         '[0-9]{2,4}-يىللىق',
         '[0-9]{1,2}-يانۋار',
         '[0-9]{1,2}-فېۋرال',
         '[0-9]{1,2}-مارت',
         '[0-9]{1,2}-ئاپرېل',
         '[0-9]{1,2}-ماي',
         '[0-9]{1,2}-ئىيۇن',
         '[0-9]{1,2}-ئىيۇل',
         '[0-9]{1,2}-ئاۋغۇست',
         '[0-9]{1,2}-سېنتەبىر',
         '[0-9]{1,2}-ئۆكتەبىر',
         '[0-9]{1,2}-نويابىر',
         '[0-9]{1,2}-دېكابىر']

    months = ['يانۋار', 'فېۋرال', 'مارت', 'ئاپرېل', 'ماي', 'ئىيۇن', 'ئىيۇل'
        , 'ئاۋغۇست', 'سېنتەبىر', 'ئۆكتەبىر', 'نويابىر', 'دېكابىر']
    time_results = []
    for i in range(len(text)):
        if re.search(r'%s' % ('|'.join(a)), text[i]) or text[i] in months:
            tem = {}
            tem['start_index'] = i
            tem['end_index'] = i + 1
            tem['entity'] = text[i]
            tem['cn_entity'] = translator(tem['entity'])
            tem['type'] = '时间'
            time_results.append(tem)

    return time_results


def get_uy_result(data):
    sentences = uy_prepare(data)
    tagger.predict(sentences)
    result = []
    for s in sentences:
        neural_result = s.to_dict(tag_type='ner')
        for entity in neural_result["entities"]:
            text_list = entity["text"].split()
            text_list[-1] = uy_segmentation(text_list[-1])[0]
            entity["no_affix"] = ' '.join(text_list)
            # entity["cn"] = translator(entity["no_affix"])
        result.append(neural_result)
    return result


CACHE_ROOT = os.path.expanduser(os.path.join('~', '.lensnlp'))

tagger = SequenceTagger.load('uy_s')
app = Flask(__name__)


@app.route('/uy/', methods=['POST', 'GET'])
def model():
    input_json = request.get_json(force=True)
    data = input_json['uy_data']

    result = get_uy_result(data)

    return Response(json.dumps(result, ensure_ascii=False), mimetype='application/json; charset=utf-8')


@app.route('/uy_ner/',methods=['GET'])
def model_1():
    data = request.args.get("uy_data")

    result = get_uy_result(data)

    return Response(json.dumps(result, ensure_ascii=False), mimetype='application/json; charset=utf-8')

if __name__ == '__main__':
    app.run(host ='0.0.0.0',port=30001,threaded=True)

