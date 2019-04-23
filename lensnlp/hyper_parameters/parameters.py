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
              'B-DATE','I-DATE','B-CAT','I-CAT','B-TITLE','I-TITLE']

id_to_label_15 = {0: "体育",1: "健康",2: "军事",3: "国际",4: "娱乐",5: "房产",6: "教育",
               7: "文化",8: "旅游",9: "时政",10: "汽车",11: "法治",12: "社会",13: "科技",14: "财经"}
