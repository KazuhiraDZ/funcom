import sys
import javalang
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf

# do NOT import keras in this header area, it will break predict.py
# instead, import keras as needed in each function

# TODO refactor this so it imports in the necessary functions
dataprep = '/scratch/funcom_bak/data/D_004'
sys.path.append(dataprep)
import Tokenizer

start = 0
end = 0

def init_tf(gpu, horovod=False):
    from keras.backend.tensorflow_backend import set_session
    
    config = tf.ConfigProto()
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = gpu

    set_session(tf.Session(config=config))

def prep(msg):
    global start
    statusout(msg)
    start = timer()

def statusout(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()

def drop():
    global start
    global end
    end = timer()
    sys.stdout.write('done, %s seconds.\n' % (round(end - start, 2)))
    sys.stdout.flush()

def index2word(tok):
	i2w = {}
	for word, index in tok.w2i.items():
		i2w[index] = word

	return i2w

def seq2sent(seq, tokenizer):
    sent = []
    check = index2word(tokenizer)
    for i in seq:
        sent.append(check[i])

    return(' '.join(sent))

def divideseqs(batchfids, seqdata, comvocabsize, tt):
    import keras.utils
    
    datseqs = list()
    comseqs = list()
    comouts = list()

    limit = -1
    c = 0
    for fid in batchfids: #seqdata['coms_%s_seqs' % (tt)].keys():

        wdatseq = seqdata['dats_%s_seqs' % (tt)][fid]
        wcomseq = seqdata['coms_%s_seqs' % (tt)][fid]

        if(len(wdatseq)<100):
            continue

        for i in range(0, len(wcomseq)):
            datseqs.append(wdatseq)

            # slice up whole comseq into seen sequence and current sequence
            # [a b c d] => [] [a], [a] [b], [a b] [c], [a b c] [d], ...
            comseq = wcomseq[0:i]
            comout = wcomseq[i]
            comout = keras.utils.to_categorical(comout, num_classes=comvocabsize)
            #print(comout)

            # extend length of comseq to expected sequence size
            # the model will be expecting all input vectors to have the same size
            for j in range(0, len(wcomseq)):
                try:
                    comseq[j]
                except IndexError as ex:
                    comseq = np.append(comseq, 0)
            #comseq = [sum(x) for x in zip(comseq, [0] * len(wcomseq))]

            comseqs.append(comseq)
            comouts.append(np.asarray(comout))

        c += 1
        if(c == limit):
            break

    datseqs = np.asarray(datseqs)
    comseqs = np.asarray(comseqs)
    comouts = np.asarray(comouts)

    return([[datseqs, comseqs], comouts])

def createbatchgen(seqdata, comvocabsize, tt, batch_size=32):
    lastbatch = 0
    allfids = list(seqdata['dats_%s_seqs' % (tt)].keys())
    # might need to sort allfids to ensure same order

    while 1:
        end = lastbatch + batch_size
        if(end > len(allfids)):
            end = len(allfids)
        batchfids = allfids[lastbatch:end]
        #print('batch: %s to %s, fids: %s to %s' % (lastbatch, end, allfids[lastbatch], allfids[end]))
        lastbatch = end

        yield divideseqs(batchfids, seqdata, comvocabsize, tt)

        if end == len(allfids):
            lastbatch = 0
            
