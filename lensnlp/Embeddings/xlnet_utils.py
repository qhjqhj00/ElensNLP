from lensnlp.Embeddings.XLNet import xlnet
from lensnlp.Embeddings.XLNet.prepro_utils import preprocess_text, encode_ids
from lensnlp.Embeddings.XLNet.classifier_utils import convert_single_example
import sentencepiece as spm
import tensorflow as tf
import collections
import os

CACHE_ROOT = os.path.expanduser(os.path.join('~', '.lensnlp'))

class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def get_xlnet_embeddings(sent):
    example = InputExample(0, sent)
    sp = spm.SentencePieceProcessor()
    sp.Load(CACHE_ROOT+'./xlnet_cased_L-24_H-1024_A-16/spiece.model')


def tokenize_fn(text):
    text = preprocess_text(text, lower=False)
    return encode_ids(sp, text)


text = 'Trump loves China'
example = InputExample(0, text)

feature = convert_single_example(5, example, None, 50, tokenize_fn)

features = collections.OrderedDict()

features["input_ids"] = tf.transpose(tf.Variable([feature.input_ids], dtype=tf.int32, name='input_ids'), [1, 0])

features["input_mask"] = tf.transpose(tf.Variable([feature.input_mask], dtype=tf.float32, name='input_mask'), [1, 0])
features["segment_ids"] = tf.transpose(tf.Variable([feature.segment_ids], dtype=tf.int32, name='segment_ids'), [1, 0])

def create_run_config(is_training, is_finetune):
    kwargs = dict(
        is_training=is_training,
        use_tpu=False,
        use_bfloat16=False,
        dropout=0.1,
        dropatt=0.1,
        init="normal",
        init_range=0.1,
        init_std=0.02,
        clamp_len=-1)

    return xlnet.RunConfig(**kwargs)

# some code omitted here...
# initialize FLAGS
# initialize instances of tf.Tensor, including input_ids, seg_ids, and input_mask

# XLNetConfig contains hyperparameters that are specific to a model checkpoint.
xlnet_config = xlnet.XLNetConfig(json_path='./xlnet_cased_L-24_H-1024_A-16/xlnet_config.json')

# RunConfig contains hyperparameters that could be different between pretraining and finetuning.
run_config = create_run_config(is_training=True, is_finetune=True)

# Construct an XLNet model
xlnet_model = xlnet.XLNetModel(
    xlnet_config=xlnet_config,
    run_config=run_config,
    input_ids=features['input_ids'],
    seg_ids=features['segment_ids'],
    input_mask=features['input_mask'])

# Get a summary of the sequence using the last hidden state
summary = xlnet_model.get_pooled_out(summary_type="last")

# Get a sequence output
seq_out = xlnet_model.get_sequence_output()

# build your applications based on `summary` or `seq_out`

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    sess.run(seq_out)

    a = seq_out.eval()
    print(seq_out.eval())
    print(seq_out.shape)