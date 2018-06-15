import pickle
import sys
import os
import math
import traceback
import argparse

import random
import tensorflow as tf
import numpy as np

seed = 1337
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

import keras
import keras.utils
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, LambdaCallback, Callback

from model import create_model
from myutils import prep, drop, divideseqs, createbatchgen, init_tf

class WeightsSaver(Callback):
    def __init__(self, model, N, outdir, modeltype):
        self.model = model
        self.N = N
        self.batch = 0
        self.outdir = outdir
        self.modeltype = modeltype

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = '%s/mdl-current-%s.h5' % (self.outdir, self.modeltype)
            self.model.save_weights(name)
        self.batch += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=str, help='0 or 1', default='0')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=400)
    parser.add_argument('--epochs', dest='epochs', type=int, default=100)
    parser.add_argument('--model-type', dest='modeltype', type=str, default='vanilla')
    parser.add_argument('--with-multigpu', dest='multigpu', action='store_true', default=False)
    parser.add_argument('--data-prep', dest='dataprep', type=str, default='../data/old')
    parser.add_argument('--outdir', dest='outdir', type=str, default='outdir')
    args = parser.parse_args()
    
    outdir = args.outdir
    dataprep = args.dataprep
    gpu = args.gpu
    batch_size = args.batch_size
    epochs = args.epochs
    modeltype = args.modeltype
    multigpu = args.multigpu

    sys.path.append(dataprep)
    import Tokenizer

    init_tf(gpu)

    prep('loading tokenizers... ')
    datstok = pickle.load(open('%s/datstokenizer.pkl' % (dataprep), 'rb'), encoding='UTF-8')
    comstok = pickle.load(open('%s/comstokenizer.pkl' % (dataprep), 'rb'), encoding='UTF-8')
    drop()

    prep('loading sequences... ')
    seqdata = pickle.load(open('%s/seqdata.pkl' % (dataprep), 'rb'))
    #seqdata = pickle.load(open('/home/cmc/pickles/seqdata.pkl', 'rb'))
    drop()
    
    steps = int(len(seqdata['coms_train_seqs'])/batch_size)+1
    datvocabsize = datstok.vocab_size
    comvocabsize = comstok.vocab_size

    print('datvocabsize %s' % (datvocabsize))
    print('comvocabsize %s' % (comvocabsize))

    # config is a dictionary so that we can send whatever info we need when creating a model
    # without having to go back and change every other model that doesn't need it
    # (e.g., batch_size for models using stateful SimpleRNN)
    config = dict()
    config['datvocabsize'] = datvocabsize
    config['comvocabsize'] = comvocabsize
    config['datlen'] = len(list(seqdata['dats_train_seqs'].values())[0])
    config['comlen'] = len(list(seqdata['coms_train_seqs'].values())[0])
    config['multigpu'] = multigpu
    config['batch_size'] = batch_size

    prep('creating model... ')
    model = create_model(modeltype, config)
    drop()

    print(model.summary())

    weightssaver = WeightsSaver(model, 50, outdir, modeltype)
    callbacks = [ weightssaver ]

    gen = createbatchgen(seqdata, comvocabsize, 'train', batch_size=batch_size)
    try:
        model.fit_generator(gen, steps_per_epoch=steps, epochs=epochs, verbose=1, max_queue_size=2, callbacks=callbacks)
    except Exception as ex:
        print(ex)
        traceback.print_exc(file=sys.stdout)

    prep('saving model... ')
    m_name = '%s/mdl-final-%s.h5' % (outdir, modeltype)
    model.save(m_name)
    drop()
