from lensnlp.utils.data_preprocess import cn_prepare,en_prepare,uy_prepare,clf_preprocess
from lensnlp.models import SequenceTagger,TextClassifier
from lensnlp.hyper_parameters import id_to_label_15
from pathlib import Path
import json
import os

import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--file', type=str, default=None)
parser.add_argument('--text', type=str, default=None)
parser.add_argument('--configure', type=str,default='configure.json')

args = parser.parse_args()
CACHE_ROOT = os.path.expanduser(os.path.join('~', '.lensnlp'))

with open(args.configure,'r') as f:
    config = json.load(f)['predict']

model_dict = {}
model_num = 0
for model_name in config:
    if config[model_name]['execute'] is True:
        model_num += 1
        language = config[model_name]['language']
        type = config[model_name]['type']
        model_path = config[model_name]['model_path']
        if type == 'ner':
            model = SequenceTagger.load(Path(CACHE_ROOT) / model_path)
        elif type == 'clf':
            model = TextClassifier.load(Path(CACHE_ROOT) / model_path)
        else:
            raise ValueError('Not Yet!')
if model_num == 0:
    raise ValueError('Specify at least one model!')


if args.file is not None:
    data = open(args.file,'r').read()
elif args.text is not None:
    data = args.text
else:
    raise ValueError('Please input text or file')

if type == 'ner':
    if language == 'cn':
        sentences = cn_prepare(data)
    elif language == 'en':
        sentences = en_prepare(data)
    elif language == 'uy':
        sentences = uy_prepare(data)
    else:
        raise ValueError('Not Yet!')
elif type == 'clf':
    sentences = clf_preprocess(data)

model.predict(sentences)

result = []
if type == 'ner':
    for s in sentences:
        neural_result = s.to_dict(tag_type='ner')
        result.append(neural_result)

elif type == 'clf':
    for s in sentences:
        result.append(id_to_label_15[int(s.labels[0].value)])

print(f'result: {result}')
