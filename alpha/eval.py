import os, sys, time, logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import pickle
import urllib.request

import pandas as pd
import scipy.misc
import numpy as np

from joblib import Parallel, delayed
from joblib.pool import has_shareable_memory

from keras.models import Model, load_model
import tensorflow as tf
from model import fun_com_model
from playground import load_embedding

def print_seq(seq, seq_name, index_word):
    print_str = seq_name + ' ' + print_seq_str(seq, seq_name, index_word)
    print(print_str)

def print_seq_str(seq, index_word):
    print_str = ''
    for word in seq:
        if word == 0: # for the padding in the dat sequences
            break
        
        print_str += str(index_word[word]) + ' '
        
    return print_str

def generate_batch(datslist_chunk, comslist_chunk, coms_vocabsize, max_caplen=50):
    dats, curs, nxts, seqs, vhists = [], [], [], [], []
    total_com_len = 0
    
    for idx in range(0, len(comslist_chunk)):
        com = comslist_chunk[idx]
        dat = datslist_chunk[idx]

        com_len = len(com)
        if(com_len == 0):
            continue

        com_len = min(com_len,max_caplen)
        total_com_len += com_len
        
        vhist = np.zeros((com_len-1, coms_vocabsize))

        for i in range(1, com_len):
            seq = np.zeros((max_caplen))
            nxt = np.zeros((coms_vocabsize))
            nxt[com[i]] = 1
            curs.append(com[i-1])
            seq[i-1] = 1
        
            if i < len(com)-1 and i < max_caplen - 1:
                vhist[i, :] = np.logical_or(vhist[i, :], vhist[i-1, :])
                vhist[i, com[i-1]] = 1
        
            nxts.append(nxt)
            dats.append(dat)
            seqs.append(seq)
        
        vhists.extend(vhist)

    logger.info('total com lens: ' + str(total_com_len))
    return np.asarray(dats), np.asarray(curs).reshape((-1, 1)), np.asarray(nxts), np.asarray(seqs), np.asarray(vhists)


if __name__ == '__main__':
    modelpath = sys.argv[1]
    
    # takes about 12.9 seconds
    t_start = time.perf_counter()
    logger.info('loading training/testing set data')
    alldata = pickle.load(open('dataprep/alldata.pkl', 'rb'))
    t_stop = time.perf_counter()
    logger.info("Elapsed time: %.1f [sec]" % (t_stop-t_start))

    t_start = time.perf_counter()
    logger.info('loading tokenizers')
    datstok = pickle.load(open('dataprep/datstokenizer.pkl', 'rb'), encoding="UTF-8")
    comstok = pickle.load(open('dataprep/comstokenizer.pkl', 'rb'), encoding="UTF-8")
    t_stop = time.perf_counter()
    logger.info("Elapsed time: %.1f [sec]" % (t_stop-t_start))
    
    logger.info('reversing tokenizer to get a mapping from index to word')
    index_word_com = {y:x for x,y in comstok.word_index.items()}
    index_word_dat = {y:x for x,y in datstok.word_index.items()}

    logger.info('preparing the sequences ...')
    dats_vocabsize = len(datstok.word_index) + 1
    coms_vocabsize = 10449 # top 10449 words are those that occur at least 100 times
    
    comslist, datslist = (list() for i in range(2))
    datslist_str, comslist_str = '', ''

    # temporary workaround
    # can't get all the test sequences in the memory (in generate_batch)
    cnt = 0
    for fid in sorted(alldata['coms_test_seqs'].keys()):
        cnt += 1
        if cnt % 1000 == 0:
            com = alldata['coms_test_seqs'][fid]
            dat = alldata['dats_test_seqs'][fid]
    
            datslist_str += print_seq_str(dat, index_word_dat) + "\n"
            comslist_str += print_seq_str(com, index_word_com) + "\n"
            
            com.append(1) # quick fix for the generate_batch which will ignore the last word of com.
            comslist.append(com)
            datslist.append(dat)

    print(datslist_str, file=open("testsrc.txt", "a"))
    print(comslist_str, file=open("testref.txt", "a"))
    
    logger.info('loop cnt: ' + str(cnt) + ', coms cnt: ' + str(len(comslist)) + ', dats cnt: ' + str(len(datslist)))
    dat, cur, nxt, seq, vhists = generate_batch(datslist, comslist, coms_vocabsize)

    t_start = time.perf_counter()
    logger.info('loading pre-trained word embedding and initializing the model')
    # copied from /scratch/codecat/codecat_final/data/embed (pretrained embedding in Alex's project)
    emb = load_embedding('../glove.codedescr.new20heldout.100d.txt', dats_vocabsize, datstok)
    max_comlen, max_datlen = 50, 50 # could call get_max_comlen() if unknown
    model = fun_com_model(coms_vocabsize=coms_vocabsize, 
                          dats_vocabsize=dats_vocabsize, 
                          max_comlen=max_comlen,
                          max_datlen=max_datlen,
                          dats_emb=emb)
    model.load_weights(modelpath)
    t_stop = time.perf_counter()
    logger.info("finished loading the model: %.1f [sec]" % (t_stop-t_start))
    
    logger.info('model predicting ...')
    probslist = model.predict([dat, cur, seq, vhists])
    logger.info('finished prediction. generated ' + str(len(probslist)) + ' sequences')

    
    pred_str = ''
    com_cnt  = 0
    word_cnt = 0
    com_len  = min(len(comslist[com_cnt]), max_comlen)
    
    for i in range(0, len(probslist) - 1):
        word_cnt += 1
        if word_cnt >= com_len:
            word_cnt   = 1
            com_cnt   += 1
            com_len    = min(len(comslist[com_cnt]), max_comlen)
            pred_str  += "\n"

        probs = probslist[i]       
        maxprob_id = np.argmax(probs)
        # maxprob    = probs.item(maxprob_id)
        pred_str  += index_word_com[maxprob_id] + ' '

    print(pred_str, file=open("predicts.txt", "a"))
