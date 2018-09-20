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
import keras.backend as K
from time import gmtime, strftime
from model import create_model
from myutils import prep, drop, batch_gen, init_tf, batch_gen_train_bleu, seq2sent
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

class mycallback(Callback):
    def __init__(self, val, steps):
        self.valgen = val
        self.refs = {}
        self.get_true()
        self.val_bleu = {}
        self.steps = steps

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        valdata = []
        data = []
        test_batch = random.randint(0, self.steps)
        fidlist, data = self.valgen[test_batch]

        data = data[0]
        batch_size = len(data)
        dats = np.array(data[0])
        comlen = len(data[1][0])
        comstart = np.zeros(comlen)
        st = comstok.w2i['<s>']
        comstart[0] = st
        coms = np.array([comstart for x in range(len(dats))])
        smls = np.array(data[2])
        for i in range(1, comlen):
            results = model.predict([dats, coms, smls], batch_size=batch_size)
            for c, s in enumerate(results):
                coms[c][i] = np.argmax(s)

        preds = []
        refs = []
        for fid, com in zip(fidlist, coms):
            pred = self.fil(seq2sent(com, comstok).split())
            if len(pred) == 0:
                continue

            try:
                refs.append(self.refs[int(fid)])
            except KeyError:
                continue
            preds.append(pred)
        self.val_bleu[epoch] = self.bleu_so_far(refs, preds)
        logs['val_bleu'] = self.val_bleu[epoch]
        print("val_bleu: {}".format(self.val_bleu[epoch]))

    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return

    def get_true(self):
        targets = open('../data/makedataset/output/coms.test', 'r')
        for line in targets:
            (fid, com) = line.split(',')
            fid = int(fid)
            com = com.split()
            com = self.fil(com)
            
            self.refs[fid] = com

    def fil(self, com):
        ret = list()
        for w in com:
            if not '<' in w:
                ret.append(w)
        return ret

    def bleu_so_far(self, refs, preds):

        try:
            Ba = corpus_bleu(refs, preds)
        except ZeroDivisionError:
            Ba = 0.0
        Ba = round(Ba * 100, 2)
        return Ba


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=str, help='0 or 1', default='0')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=200)
    parser.add_argument('--epochs', dest='epochs', type=int, default=100)
    parser.add_argument('--model-type', dest='modeltype', type=str, default='vanilla')
    parser.add_argument('--with-multigpu', dest='multigpu', action='store_true', default=False)
    parser.add_argument('--data', dest='dataprep', type=str, default='/scratch/funcom/data/standard')
    parser.add_argument('--outdir', dest='outdir', type=str, default='/scratch/funcom/data/outdir/models')
    parser.add_argument('--dtype', type=str, default='float32')
    dset = parser.add_mutually_exclusive_group()
    dset.add_argument('--challenge', dest='challenge', action='store_true', default=False)
    dset.add_argument('--obfuscate', dest='obf', action='store_true', default=False)
    dset.add_argument('--sbt', dest='sbt', action='store_true', default=False)
    dset.add_argument('--name', dest='name', action='store_true', default=False)
    dset.add_argument('--body', dest='body', action='store_true', default=False)
    dset.add_argument('--sig', dest='sig', action='store_true', default=False)
    dset.add_argument('--stand', dest='stand', action='store_true', default=False)
    dset.add_argument('--septs', dest='septs', action='store_true', default=False)
    dset.add_argument('--3d-sdats', dest='threed', action='store_true', default=False)
    args = parser.parse_args()
    
    outdir = args.outdir
    dataprep = args.dataprep
    gpu = args.gpu
    batch_size = args.batch_size
    epochs = args.epochs
    modeltype = args.modeltype
    multigpu = args.multigpu
    challenge = args.challenge
    obf = args.obf
    sbt = args.sbt
    body = args.body
    name = args.name
    sig = args.sig
    stand = args.stand
    septs = args.septs
    threed = args.threed

    K.set_floatx(args.dtype)

    sys.path.append(dataprep)
    import tokenizer

    init_tf(gpu)
    modelname = modeltype
    if dataprep == '../data/makedataset' and challenge:
        dataprep = '../data/standard'
        modelname = 'chalzerodats_'+modeltype

    elif dataprep == '../data/makedataset' and obf:
        dataprep = '../data/obfuscation'
        modelname = 'obf_'+modeltype
    elif dataprep == '../data/makedataset' and sbt:
        dataprep = '../data/sbt'
        modelname = 'sbt_'+modeltype

    elif dataprep == '../data/makedataset' and name:
        dataprep = '../data/name_only'
        modelname = 'nameonly_'+modeltype

    elif dataprep == '../data/makedataset' and body:
        dataprep = '../data/body_only'
        modelname = 'bodyonly_'+modeltype

    elif dataprep == '../data/makedataset' and sig:
        dataprep = '../data/sig_only'
        modelname = 'sigonly_'+modeltype

    elif dataprep == '../data/makedataset' and stand:
        dataprep = '../data/standard'
        modelname = 'standard_'+modeltype

    #elif septs:
    #    dataprep = '/scratch/funcom/data/standard_filecontext'
    #    modelname = 'septs_'+modeltype

    elif threed:
        modelname = 'threed_'+modeltype

    prep('loading tokenizers... ')
    tdatstok = pickle.load(open('%s/tdats.tok' % (dataprep), 'rb'), encoding='UTF-8')
    sdatstok = pickle.load(open('%s/sdats.tok' % (dataprep), 'rb'), encoding='UTF-8')
    comstok = pickle.load(open('%s/coms.tok' % (dataprep), 'rb'), encoding='UTF-8')
    if not sbt:
        smltok = pickle.load(open('%s/smls.tok' % (dataprep), 'rb'), encoding='UTF-8')
    drop()

    prep('loading sequences... ')
    seqdata = pickle.load(open('%s/dataset.pkl' % (dataprep), 'rb'))
    drop()

    steps = int(len(seqdata['ctrain'])/batch_size)+1
    valsteps = int(len(seqdata['cval'])/100)+1
    
    tdatvocabsize = tdatstok.vocab_size
    comvocabsize = comstok.vocab_size
    
    if not sbt:
        smlvocabsize = smltok.vocab_size

    print('tdatvocabsize %s' % (tdatvocabsize))
    print('comvocabsize %s' % (comvocabsize))
    if not sbt:
        print('smlvocabsize %s' % (smlvocabsize))
    print('batch size {}'.format(batch_size))
    print('steps {}'.format(steps))
    print('training data size {}'.format(steps*batch_size))
    print('vaidation data size {}'.format(valsteps*100))

    if challenge:
        print('training/validating on challenge set data')
    else:
        print('training/validating on verb first data')

    print('------------------------------------------')

    config = dict()
    config['tdatvocabsize'] = tdatvocabsize
    config['comvocabsize'] = comvocabsize
    if not sbt:
        config['smlvocabsize'] = smlvocabsize

    config['tdatlen'] = len(list(seqdata['dttrain'].values())[0])
    config['sdatlen'] = seqdata['config']['sdatlen']


    config['comlen'] = len(list(seqdata['ctrain'].values())[0])
    if not sbt:
        config['smllen'] = len(list(seqdata['strain'].values())[0])
    config['multigpu'] = multigpu
    config['batch_size'] = batch_size

    prep('creating model... ')
    config, model = create_model(modeltype, config)
    drop()

    print(model.summary())
    

    gen = batch_gen(seqdata, comvocabsize, 'train', modeltype, config['num_input'], config, batch_size=batch_size, threed=threed)
    #checkpoint = ModelCheckpoint(outdir+'/'+modeltype+'_E{epoch:02d}_TA{acc:.2f}_VA{val_acc:.2f}_VB{val_bleu:}.h5', monitor='val_loss')
    checkpoint = ModelCheckpoint(outdir+'/'+modelname+'_E{epoch:02d}.h5')
    valgen = batch_gen(seqdata, comvocabsize, 'val', modeltype, config['num_input'], config, batch_size=batch_size, threed=threed)

    # If you want it to calculate BLEU Score after each epoch use callback_valgen and test_cb
    #####
    #callback_valgen = batch_gen_train_bleu(seqdata, comvocabsize, 'val', modeltype, batch_size=batch_size)
    #test_cb = mycallback(callback_valgen, steps)
    #####
    callbacks = [ checkpoint ]

    try:
        model.fit_generator(gen, steps_per_epoch=steps, epochs=epochs, verbose=1, max_queue_size=2, callbacks=callbacks, validation_data=valgen, validation_steps=valsteps)# 
    except Exception as ex:
        print(ex)
        traceback.print_exc(file=sys.stdout)

    prep('saving model... ')
    m_name = '%s/mdl-final-%s.h5' % (outdir, modelname)
    model.save(m_name)
    drop()
