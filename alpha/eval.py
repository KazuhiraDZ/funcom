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

import configparser, argparse

def print_seq(seq, seq_name, index_word):
    print_str = seq_name + ' ' + print_seq_str(seq, index_word)
    print(print_str)

def print_seq_str(seq, index_word):
    print_str = ''
    for word in seq:
        if word == 0: # for the padding in the dat sequences
            break
        
        print_str += str(index_word[word]) + ' '
        
    return print_str

def beam_search(predict, source_sequence, coms_vocabsize, max_caplen, sos, eos, k):
    dead_k = 0 # samples that reached eos
    dead_samples = []
    dead_scores = []

    live_k = 1 # samples that did not yet reached eos
    live_preds = [[]]
    seq = np.zeros((max_caplen))
    curs = [sos] # as a feed in the beginning (temporary workaround)
    vhists = np.zeros((1, coms_vocabsize))
    live_inputs = [np.asarray([source_sequence]),
                   np.asarray(curs).reshape((-1, 1)),
                   np.asarray([seq]),
                   np.asarray(vhists)]
    live_scores = [0]
    word_cnt =  0

    while live_k and dead_k < k:
        probslist = predict(live_inputs)
        # for each row in live_score, compute #coms_vocabsize scores for each word in probslist
        cand_scores = np.array(live_scores)[:,None] - np.log(probslist)
        
        # pick the best scores
        cand_flat = cand_scores.flatten()
        ranks_flat = np.argpartition(cand_flat, k)[:k]
        live_scores = cand_flat[ranks_flat]
        live_preds = [live_preds[r//coms_vocabsize]+[r%coms_vocabsize] for r in ranks_flat]
        
        # update live_preds, dead_samples
        zombie = [s[-1] == eos or len(s) >= max_caplen for s in live_preds]
        dead_samples += [s for s,z in zip(live_preds,zombie) if z]
        dead_scores += [s for s,z in zip(live_scores,zombie) if z]
        dead_k = len(dead_samples)
        live_preds = [s for s,z in zip(live_preds,zombie) if not z]
        live_scores = [s for s,z in zip(live_scores,zombie) if not z]
        live_k = len(live_preds)

        # update live_inputs
        seq = np.zeros((max_caplen))
        seq[word_cnt] = 1
        new_vhists = np.zeros((len(ranks_flat), coms_vocabsize))
        for r in ranks_flat:
            new_vhists[r//coms_vocabsize][r%coms_vocabsize] = 1
            
        vhists = [np.logical_or(vhists[i//coms_vocabsize], j) for i,j in zip(ranks_flat,new_vhists)]
        vhists = [s for s,z in zip(vhists,zombie) if not z]
        curs = [r%coms_vocabsize for r in ranks_flat]
        curs = [s for s,z in zip(curs,zombie) if not z]
        
        live_inputs = [np.asarray([source_sequence for i in range(live_k)]),
                       np.asarray(curs).reshape((-1, 1)),
                       np.asarray([seq for i in range(live_k)]),
                       np.asarray(vhists)]

        word_cnt += 1
        # end prediction

    return dead_samples[np.argmax(dead_scores)]

    
## greedy search
def greedy_search(predict, source_sequence, coms_vocabsize, max_caplen, sos, eos):
    search_logger = logging.getLogger(__name__ + ': greedy_search')
    
    prediction = []
    word_cnt = 0
    pred = -1
    vhist = np.zeros((1, coms_vocabsize))
    
    cur = sos
    seq = np.zeros((max_caplen))
    while True:
        # predict a word
        probslist = predict([np.asarray([source_sequence]),
                             np.asarray(cur).reshape((-1, 1)),
                             np.asarray([seq]),
                             np.asarray(vhist)])
        
        probs = probslist[0]
        pred = np.argmax(probs)
        prediction.append(pred)
        # maxprob = probs.item(pred)
        # search_logger.info(str(word_cnt) + ': wordid#' + str(pred))

        word_cnt += 1
        if word_cnt >= max_caplen or pred == eos:
            break

        # prepare for the next iteration
        seq = np.zeros((max_caplen))
        seq[word_cnt-1] = 1
        cur = pred # for next prediction
        vhist[0, pred] = 1
        # end prediction

    return prediction

def parse_config_var(config, var_name, var_default):
    var_val = var_default
    if config['DEFAULT'][var_name]:
        var_val = config['DEFAULT'][var_name]
        logger.info("test.ini: " + var_name + "=" + var_val)
    else:
        logger.warning("test.ini: no " + var_name + " configured. using: " + var_val)

    return var_val


def parse_config(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)
    
    dataprep='./dataprep'
    # copied from /scratch/codecat/codecat_final/data/embed (pretrained embedding in Alex's project)
    embfile='../glove.codedescr.new20heldout.100d.txt'
    outdir='./'

    dataprep = parse_config_var(config, 'dataprep', dataprep)
    embfile  = parse_config_var(config, 'embfile', embfile)
    outdir   = parse_config_var(config, 'outdir', outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    return {'dataprep': dataprep,
            'embfile' : embfile,
            'outdir'  : outdir,}
    
def parse_args():
    parser = argparse.ArgumentParser(description='Run models to predict comments.')
    parser.add_argument('--config', nargs=1, help='the config file, see test.ini as an example', required=True)
    parser.add_argument('--beamsearch', metavar='K', type=int, nargs=1, help='use beam search, and set K as beam width; if not used, run greedy search')
    parser.add_argument('model', nargs='+', help='a model\'s file path')
    args = parser.parse_args()
    modelpath = args.model[0]
    configfile = args.config[0]
    beamsearch = -1
        
    logger.info("input model: " + modelpath)
    logger.info("config file: " + configfile)
    
    if args.beamsearch:
        beamsearch = args.beamsearch[0]
        logger.info("beamsearch, k=" + str(beamsearch))
    else:
        logger.info("greedysearch")

    return {'modelpath':modelpath,
            'configfile':configfile,
            'beamsearch':beamsearch,}

    
if __name__ == '__main__':
    args = parse_args()
    modelpath  = args['modelpath']
    beamwidth = args['beamsearch']
    search = ''
    if beamwidth == -1:
        search = 'greedy'
    else:
        search = 'beam'
    
    config     = parse_config(args['configfile'])
    dataprep   = config['dataprep']
    embfile    = config['embfile']
    outdir     = config['outdir']
    timestr    = time.strftime("%Y%m%d-%H%M%S")
    outputfile = {'srcfile':os.path.join(outdir, 'testsrc') + timestr + '.txt',
                  'reffile':os.path.join(outdir, 'testref') + timestr + '.txt',
                  'predict':os.path.join(outdir, 'predicts') + timestr + '.txt',}
    
    ### loading the data and parameters for the model
    ###
    # takes about 12.9 seconds
    t_start = time.perf_counter()
    logger.info('loading training and testing set data...')
    alldata = pickle.load(open(os.path.join(dataprep, 'alldata.pkl'), 'rb'))
    t_stop = time.perf_counter()
    logger.info("finished loading data sets: %.1f [sec]" % (t_stop-t_start))

    t_start = time.perf_counter()
    logger.info('loading tokenizers...')
    datstok = pickle.load(open(os.path.join(dataprep, 'datstokenizer.pkl'), 'rb'), encoding="UTF-8")
    comstok = pickle.load(open(os.path.join(dataprep, 'comstokenizer.pkl'), 'rb'), encoding="UTF-8")
    t_stop = time.perf_counter()
    logger.info("finished loading tokenizers: %.1f [sec]" % (t_stop-t_start))
    
    dats_vocabsize = len(datstok.word_index) + 1
    coms_vocabsize = 10449 # top 10449 words are those that occur at least 100 times
    
    max_comlen, max_datlen = 50, 50
    ###
    ### End: loading the data and parameters
    
    logger.info('reversing tokenizer to get a mapping from index to word')
    index_word_com = {y:x for x,y in comstok.word_index.items()}
    index_word_dat = {y:x for x,y in datstok.word_index.items()}

    logger.info('preparing the sequences ...')
    comslist, datslist = (list() for i in range(2))
    datslist_str, comslist_str = '', ''

    cnt = 0
    for fid in sorted(alldata['coms_test_seqs'].keys()):
        cnt += 1
        com = alldata['coms_test_seqs'][fid]
        dat = alldata['dats_test_seqs'][fid]
        
        comslist.append(com)
        datslist.append(dat)

        datslist_str += print_seq_str(dat, index_word_dat) + "\n"
        comslist_str += print_seq_str(com, index_word_com) + "\n"

    
    print(datslist_str, file=open(outputfile['srcfile'], "w"))
    print(comslist_str, file=open(outputfile['reffile'], "w"))
    
    logger.info('loop cnt: ' + str(cnt) + ', coms cnt: ' + str(len(comslist)) + ', dats cnt: ' + str(len(datslist)))
    
    t_start = time.perf_counter()
    logger.info("loading pre-trained word embedding...")
    emb = load_embedding(embfile, dats_vocabsize, datstok)
    t_stop = time.perf_counter()
    logger.info("finished loading the embedding: %.1f [sec]" % (t_stop-t_start))
    
    t_start = time.perf_counter()
    logger.info("initializing the model...")
    model = fun_com_model(coms_vocabsize=coms_vocabsize, 
                          dats_vocabsize=dats_vocabsize, 
                          max_comlen=max_comlen,
                          max_datlen=max_datlen,
                          dats_emb=emb)
    model.load_weights(modelpath)
    t_stop = time.perf_counter()
    logger.info("finished initializing the model: %.1f [sec]" % (t_stop-t_start))
    
    try:
        sos = comstok.word_index['<s>']
    except:
        logger.warning("cannot find <s> in the comment dictionary")
        sos = -1

    try:    
        eos = comstok.word_index['</s>']
    except:
        logger.warning("cannot find </s> in the comment dictionary")
        eos = 0

    logger.info('model predicting ...')
    predlist = []
    
    for i in range(len(datslist)):
        if sos == -1:
            sos = comslist[i][0]

        if search == 'greedy':
            prediction = greedy_search(model.predict, datslist[i], coms_vocabsize, max_comlen, sos, eos)
        elif search == 'beam':
            logger.info('beam search...')
            prediction = beam_search(model.predict, datslist[i], coms_vocabsize, max_comlen, sos, eos, beamwidth)
        else:
            logger.error("cannot recognize search algorithm: " + search + ", exit.")
            sys.exit()
        
        prediction_str = print_seq_str(prediction, index_word_com) + "\n"
        logger.info(str(i+1) + '/' + str(len(datslist)) + ' prediction: ' + prediction_str)
        predlist.append(prediction)

    logger.info('finished prediction. generated ' + str(len(predlist)) + ' sequences')
    prediction_str = "\n".join([print_seq_str(prediction, index_word_com) for prediction in predlist])
    print(prediction_str, file=open(outputfile['predict'], "w"))
