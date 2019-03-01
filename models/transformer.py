from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, Bidirectional, CuDNNGRU, RepeatVector, Permute, TimeDistributed, dot
from keras.optimizers import RMSprop, Adamax, Adam
import keras
import keras.utils
import tensorflow as tf
from keras import metrics

from keras_transformer import get_model
#from models.libtransformer import Transformer, LRSchedulerPerStep, LRSchedulerPerEpoch

import numpy as np

seed = 1337
np.random.seed(seed)

## WARNING This is still buggy!!

class TransformerModel:
    def __init__(self, config):
        
        # override default tdatlen
        config['tdatlen'] = 50
        
        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.datlen = config['tdatlen']
        self.comlen = config['comlen']
        
        self.embdims = 100
        self.recdims = 256

        self.config['num_input'] = 2
        self.config['num_output'] = 1

    def create_model(self):
        
        #dat_input = Input(shape=(self.datlen,))
        #com_input = Input(shape=(self.comlen,))

#        trans = Transformer(self.tdatvocabsize, self.comvocabsize, len_limit=70, d_model=512, d_inner_hid=512, \
#            n_head=8, d_k=64, d_v=64, layers=2, dropout=0.1)

#        optimizer = Adam(0.001, 0.9, 0.98, epsilon=1e-9)
#        trans.compile(optimizer=optimizer)

        model = get_model(
                    token_num=[self.tdatvocabsize, self.comvocabsize],
                    embed_dim=30,
                    encoder_num=3,
                    decoder_num=2,
                    head_num=3,
                    hidden_dim=120,
                    attention_activation='relu',
                    feed_forward_activation='relu',
                    dropout_rate=0.05,
                    use_same_embed=False,
                    embed_weights=[np.random.random((self.tdatvocabsize, 30)),
                                   np.random.random((self.comvocabsize, 30))],
                    embed_trainable=[True,True],
                )

        model.compile(
                    optimizer=keras.optimizers.Adam(),
                    loss=keras.losses.sparse_categorical_crossentropy,
                    metrics={},
                    # Note: There is a bug in keras versions 2.2.3 and 2.2.4 which causes "Incompatible shapes" error, if any type of accuracy metric is used along with sparse_categorical_crossentropy. Use keras<=2.2.2 to use get validation accuracy.
                )

        if self.config['multigpu']:
            model = keras.utils.multi_gpu_model(model, gpus=2)

        return self.config, model
