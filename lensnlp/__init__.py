from . import utils
from . import models
from . import trainers
from . import hyper_parameters
from . import embeddings
from . import modules

import logging.config
from .predict import ner,clf
from .train import cls_train, seq_train

import os

CACHE_ROOT = os.path.expanduser(os.path.join('~', '.lensnlp'))

if not os.path.exists(CACHE_ROOT):
    os.mkdir(CACHE_ROOT)

__version__ = "0.0.0"

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)-15s %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
    },
    'loggers': {
        'lensnlp': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False
        }
    },
    'root': {
        'handlers': ['console'],
        'level': 'WARNING'
    }
})

logger = logging.getLogger('lensnlp')


