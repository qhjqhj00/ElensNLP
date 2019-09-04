import os
import pickle
import logging

log = logging.getLogger('lensnlp')

CACHE_ROOT = os.path.expanduser(os.path.join('~', '.lensnlp'))

try:
    with open(CACHE_ROOT+'/four_corner.pkl', 'rb') as f:
        four_corner_dic = pickle.load(f)
except:
    log.info('4c mode is not supported.')
