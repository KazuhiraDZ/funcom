import pickle
from joblib import Parallel, delayed
import numpy as np
import random
import tensorflow as tf
import time, logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
    
seed = 1337
np.random.seed(1337)
random.seed(1337)
tf.set_random_seed(seed)

import keras
from keras.preprocessing.text import Tokenizer
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
#config.gpu_options.visible_device_list = gpu
set_session(tf.Session(config=config))

import utils
from model import fun_com_model


def gen_batch_in_thread(datslist, comslist, coms_vocabsize, n_jobs=4, size_per_thread=32):
    dats, curs, nxts, seqs, vhists = [], [], [], [], []
    returns = Parallel(n_jobs=4, backend='threading')(
                            delayed(generate_batch)
                            (datslist, comslist, coms_vocabsize, size=size_per_thread) for i in range(0, n_jobs))

    for triple in returns:
        dats.extend(triple[0])
        curs.extend(triple[1])
        nxts.extend(triple[2])
        seqs.extend(triple[3])
        vhists.extend(triple[4])

    newdats = []
    for dat in dats:
        newdats.append(np.asarray(dat))
    dats = np.asarray(newdats)

    return dats, np.asarray(curs).reshape((-1,1)), np.asarray(nxts), np.asarray(seqs), np.asarray(vhists)

def generate_batch(datslist, comslist, coms_vocabsize, size=32, max_caplen=50):
    dats, curs, nxts, seqs, vhists = [], [], [], [], []
    
    for idx in np.random.randint(len(comslist), size=size):
        # logger.info('idx: ' + str(idx))
        com = comslist[idx]
        dat = datslist[idx]
        
        if(len(com) == 0):
            continue
        
        vhist = np.zeros((len(com)-1, coms_vocabsize))

        for i in range(1, len(com)):
            seq = np.zeros((max_caplen))
            nxt = np.zeros((coms_vocabsize))
            nxt[com[i]] = 1     
            curs.append(com[i-1])
            seq[i-1] = 1

            if i < len(com)-1:
                vhist[i, :] = np.logical_or(vhist[i, :], vhist[i-1, :])
                vhist[i, com[i-1]] = 1

            nxts.append(nxt)
            dats.append(dat)
            seqs.append(seq)

        vhists.extend(vhist)

    return dats, curs, nxts, seqs, vhists

def load_embedding(embfile, vocab_size, tokenizer):

    ei = dict()

    st = time.perf_counter()
    logger.info('reading the embedding file: ' + embfile)
    f = open(embfile)
    for line in f:
        values = line.split()
        word = values[0]
        value = np.asarray(values[1:], dtype='float32')
        ei[word] = value
    f.close()
    et = time.perf_counter()
    logger.info("Elapsed time: %.1f [sec]" % (et - st))
    
    st = time.perf_counter()
    logger.info('creating a weight matrix for words in training docs')
    embedded_dims = 100
    embed_matrix = np.zeros((vocab_size, embedded_dims))
    for word, i in tokenizer.word_index.items():
            embedding_vector = ei.get(word)
            if embedding_vector is not None:
                    embed_matrix[i] = embedding_vector
                    
    print('loaded %s word vectors' % len(ei))
    et = time.perf_counter()
    logger.info("Elapsed time: %.1f [sec]" % (et - st))
    
    return embed_matrix

if __name__ == '__main__':

    print('loading training/testing set data')

    # takes about 22 seconds
    alldata = pickle.load(open('dataprep/alldata.pkl', 'rb'))

    print('loading tokenizers')

    datstok = pickle.load(open('dataprep/datstokenizer.pkl', 'rb'), encoding="UTF-8")
    comstok = pickle.load(open('dataprep/comstokenizer.pkl', 'rb'), encoding="UTF-8")

    dats_vocabsize = len(datstok.word_index) + 1
    coms_vocabsize = 10449 # top 10449 words are those that occur at least 100 times
    
    print('dats_vocabsize: %d' % dats_vocabsize)
    print('coms_vocabsize: %d' % coms_vocabsize)

    max_comlen = 50 # could call get_max_comlen() if unknown
    max_datlen = 50

    print('loading pre-trained word embedding')

    # copied from /scratch/codecat/codecat_final/data/embed
    emb = load_embedding('../glove.codedescr.new20heldout.100d.txt', dats_vocabsize, datstok)

    print('creating model')

    model = fun_com_model(coms_vocabsize=coms_vocabsize, 
                          dats_vocabsize=dats_vocabsize, 
                          max_comlen=max_comlen,
                          max_datlen=max_datlen,
                          dats_emb=emb)

    print(model.summary())
    
    comslist = list()
    datslist = list()
    
    for fid in sorted(alldata['coms_train_seqs'].keys()):
        com = alldata['coms_train_seqs'][fid]
        dat = alldata['dats_train_seqs'][fid]
        
        comslist.append(com)
        datslist.append(dat)

    hist_path = 'history/'
    mdl_path = 'models/'
    batch_num = 70

    n_jobs = 8
    size_per_thread = 64
    batch_size = n_jobs * size_per_thread
    
    hist_loss = []

    for i in range(0, 100):
        for j in range(0, batch_num):            
            # st = datetime.now()
            st = time.perf_counter()

            dat1, cur1, nxt1, seq1, vhists1 = gen_batch_in_thread(datslist, comslist, coms_vocabsize, n_jobs=n_jobs, size_per_thread=size_per_thread)
            
            dat2, cur2, nxt2, seq2, vhists2 = gen_batch_in_thread(datslist, comslist, coms_vocabsize, n_jobs=4, size_per_thread=16)

            #print('dat1 ' + str(dat1.shape))  
            #print('cur1 ' + str(cur1.shape))  
            #print('nxt1 ' + str(nxt1.shape))    
            #print('seq1 ' + str(seq1.shape))    
            #print('vhs1 ' + str(vhists1.shape)) 
            
            hist = model.fit([dat1, cur1, seq1, vhists1], nxt1, batch_size=batch_size, epochs=1, verbose=0,
                                            validation_data=([dat2, cur2, seq2, vhists2], nxt2), shuffle=True)

            # et = datetime.now()
            et = time.perf_counter()
            # samplespersecond = (et.seconds - st.seconds) / batch_size
            samplespersecond = (et - st) / batch_size

            print("epoch {0}, batch {1} - training loss : {2}, validation loss: {3}, samples/second: {4}"
                        .format(i, j, hist.history['loss'][-1], hist.history['val_loss'][-1], samplespersecond))
            
            hist_loss.extend(hist.history['loss'])

        print('check point')
        m_name = "{0}{1}_{2}.h5".format(mdl_path, i, time.time())
        model.save_weights(m_name)
        pickle.dump({'loss':hist_loss}, open(hist_path+ 'history.pkl', 'wb'))
