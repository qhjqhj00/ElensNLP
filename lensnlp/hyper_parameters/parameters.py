import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from collections import OrderedDict

Parameter = OrderedDict()
Parameter['lr'] = 0.1
Parameter['anneal_rate'] = 0.5
Parameter['patience'] = 3
Parameter['batch_size'] = 32
Parameter['max_epoch'] = 100
Parameter['crf'] = True
Parameter['dropout'] = 0.0
Parameter['word_dropout'] = 0.05
Parameter['locked_dropout'] = 0.5
Parameter['rnn'] = True
Parameter['rnn_layer'] = 1
Parameter['rnn_type'] = 'LSTM'
Parameter['bilstm'] = True
Parameter['anneal_against_train'] = False
Parameter['train_with_dev'] = True
Parameter['cache_path'] = '/mnt/flair_cache/'

tag_filter = ['B-LOC', 'B-ORG', 'B-PER', 'I-LOC', 'I-ORG', 'I-PER', 'O','B-MOV',
              'I-MOV','B-Time','I-Time','B-NUM','I-NUM','B-MEA','I-MEA','B-PHO',
              'I-PHO','B-WEB','I-WEB','B-POST','I-POST','B-TIME','I-TIME','B-MONEY','I-MONEY',
              'B-DATE','I-DATE','B-CAT','I-CAT','B-TITLE','I-TITLE','B-MISC','I-MISC']

cn_clf = {0: "体育",1: "健康",2: "军事",3: "国际",4: "娱乐",5: "房产",6: "教育",
               7: "文化",8: "旅游",9: "时政",10: "汽车",11: "法治",12: "社会",13: "科技",14: "财经"}

en_clf = {0: 'Company', 1: 'EducationalInstitution', 2: 'Artist',3: 'Athlete', 4: 'OfficeHolder',
          5: 'MeanOfTransportation', 6: 'Building', 7: 'NaturalPlace', 8: 'Village', 9: 'Animal',
          10: 'Plant', 11: 'Album', 12: 'Film', 13: 'WrittenWork'}

uy_clf = {0:'文化', 1:'体育', 2:'其他',3: '读物',4: '生活',5: '新疆',6: '国际',7: '国内'}

emo = {-1: '负面',0: '中性',1: '正面'}

