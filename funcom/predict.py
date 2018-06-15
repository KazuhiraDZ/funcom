import os
import sys
import traceback
import pickle
import argparse
import collections

import random
import tensorflow as tf
import numpy as np

seed = 1337
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, as_completed
import multiprocessing
from itertools import product

from timeit import default_timer as timer

from model import create_model
from myutils import prep, drop, statusout, divideseqs, createbatchgen, seq2sent, index2word, init_tf

def gendescr(model, dat, comstok, comlen, strat='greedy'):
    # right now, only greedy search is supported...
    st = comstok.w2i['<s>']
    
    pred = np.zeros(comlen)
    pred[0] = st
    
    if(strat == 'greedy'):
        for i in range(1, comlen):
            t = model.predict([np.asarray([dat]), np.asarray([pred])], batch_size=1)
            t = np.argmax(t)
            pred[i] = t

    return(seq2sent(pred, comstok))

def gendescrs(wrkunits):
    rets = list()
    
    import keras
    from keras.models import load_model
    
    model = create_model(modeltype, datvocabsize, comvocabsize)
    model.load_weights('%s/mdl-current-%s.h5' % (outdir, modeltype))
    
    global comstok
    
    for wrkunit in wrkunits:
        (fid, dat, comlen) = wrkunit
        rets.append((fid, gendescr(model, dat, comstok, comlen)))
    
    return(rets)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model-type', dest='modeltype', type=str, default='vanilla-lstm')
    parser.add_argument('--num-procs', dest='numprocs', type=int, default='4')
    parser.add_argument('--gpu', dest='gpu', type=str, default='')
    args = parser.parse_args()

    outdir = 'outdir'
    dataprep = '/scratch/funcom/data/D_004'
    global modeltype
    modeltype= args.modeltype
    numprocs = args.numprocs
    gpu = args.gpu
    
    # predict.py typically uses the CPU
    # the reason is that we need to call model.predict() sequentially, because
    # the model outputs predictions for each word, and to create a sentence
    # we need to predict each word at a time
    # however... we do need the GPU for models that use CuDNNGRU
    # if this is the case, it will probably only work with numprocs=1
    
    if not gpu == '':
        numprocs = 1
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    sys.path.append(dataprep)
    import Tokenizer

    prep('loading tokenizers... ')
    global datstok
    global comstok
    datstok = pickle.load(open('%s/datstokenizer.pkl' % (dataprep), 'rb'), encoding='UTF-8')
    comstok = pickle.load(open('%s/comstokenizer.pkl' % (dataprep), 'rb'), encoding='UTF-8')
    drop()

    prep('loading sequences... ')
    seqdata = pickle.load(open('%s/seqdata.pkl' % (dataprep), 'rb'))
    drop()

    allfids = list(seqdata['coms_test_seqs'].keys())
    global datvocabsize
    global comvocabsize
    datvocabsize = datstok.vocab_size
    comvocabsize = comstok.vocab_size

    prep('computing predictions... ')
    predsfile = open('%s/predict-%s.txt' % (outdir, modeltype), 'w')
    st = timer()
    wrkunits = collections.defaultdict(list)
    
    multiprocessing.log_to_stderr()
    pool = ProcessPoolExecutor(max_workers=numprocs)
    
    for c, fid in enumerate(allfids):
        dat = seqdata['dats_test_seqs'][fid]
        comlen = len(seqdata['coms_test_seqs'][fid]) # should be fixed size anyway
        
        r = random.randint(0, numprocs)
        wrkunits[r].append((fid, dat, comlen))

        if(c > 0 and c % 1000 == 0):
            
            for rets in pool.map(gendescrs, wrkunits.values()):
                #for rets in allrets:
                    for ret in rets:
                        (fid, pred) = ret
                        predsfile.write('%s\t%s\n' % (fid, pred))
                        predsfile.flush()
            
            wrkunits = collections.defaultdict(list)
            
            et = timer()
            statusout('%s/%ss, ' % (c, round(et-st, 1)))
            st = timer()
        
    predsfile.close()
    drop()

    #batch_size = 1800
    #steps = int(len(seqdata['coms_test_seqs'])/batch_size)+1

    #gen = createbatchgen(seqdata, comvocabsize, 'test', batch_size=batch_size)
    #try:
    #    score = model.evaluate_generator(gen, steps=steps, verbose=1, max_queue_size=2)
    #    print('loss: %s, accuracy: %s' % (score[0], score[1]))
    #except Exception as ex:
    #    print(ex)
    #    traceback.print_exc(file=sys.stdout)


