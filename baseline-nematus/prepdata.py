import time, logging, sys, os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import pickle
import json
import operator
import random
random.seed(1337)

import keras
from keras.preprocessing.text import Tokenizer

import configparser, argparse

def createvocab( origvocab ):
    vocab = dict()
    vocab['eos'] = 0
    vocab['UNK'] = 1
    
    i = 2
    for word, count in origvocab.items():
        if word == 'eos' or word == 'UNK':
            continue
        
        vocab[word] = i
        i += 1

    return vocab


def vocabfiles( srctok, tgttok, srcoutfile, tgtoutfile ):
    src_vocabsize = len(srctok.word_index) + 1
    tgt_vocabsize = 10449 # top 10449 words are those that occur at least 100 times

    print("vocab sizes: %s (src) %s (tgt) " % (src_vocabsize, tgt_vocabsize))
    
    srcvocab = srctok.word_index # Tokenizer prepared by Keras should have been sorted by the word counts (in the reversed order)
    tgtvocab = tgttok.word_index

    srcvocab = createvocab(srcvocab)
    tgtvocab = createvocab(tgtvocab)
    
    srcout = open( srcoutfile, 'w')
    json.dump(srcvocab, srcout, indent=2, ensure_ascii=False)
    srcout.close()    

    tgtout = open( tgtoutfile, 'w')
    json.dump(tgtvocab, tgtout, indent=2, ensure_ascii=False)
    tgtout.close()

    
def print_seq_str(seq, index_word):
    print_str = ''
    for word in seq:
        if word == 0: # for the padding in the dat sequences
            break
        
        print_str += str(index_word[word]) + ' '
        
    return print_str


def getdata_from_alldata(alldata, field_src, field_tgt, index_word_src, index_word_tgt, srcoutfile, tgtoutfile):
    srclist = list()
    tgtlist = list()
    
    for fid in sorted(alldata[field_src].keys()):
        src = alldata[field_src][fid]
        src_str = print_seq_str(src, index_word_src)
        srclist.append(src_str)

        tgt = alldata[field_tgt][fid]
        tgt_str = print_seq_str(tgt, index_word_tgt)
        tgtlist.append(tgt_str)

    with open(srcoutfile, mode='wt', encoding='utf-8') as outf:
        outf.write('\n'.join(srclist))

    with open(tgtoutfile, mode='wt', encoding='utf-8') as outf:
        outf.write('\n'.join(tgtlist))


def validfiles(alldata, index_word_src, index_word_tgt, srcoutfile, tgtoutfile):
    srclist = list()
    tgtlist = list()

    keys = sorted(alldata['coms_train_seqs'].keys())
    random.shuffle(keys)

    cnt = 0
    for fid in keys:
        cnt += 1
        src = alldata['dats_train_seqs'][fid]
        src_str = print_seq_str(src, index_word_src)
        srclist.append(src_str)

        tgt = alldata['coms_train_seqs'][fid]
        tgt_str = print_seq_str(tgt, index_word_tgt)
        tgtlist.append(tgt_str)

        if cnt >= 3000:
            break

    with open(srcoutfile, mode='wt', encoding='utf-8') as outf:
        outf.write('\n'.join(srclist))
        outf.write('\n')

    with open(tgtoutfile, mode='wt', encoding='utf-8') as outf:
        outf.write('\n'.join(tgtlist))
        outf.write('\n')

        
def parse_args():
    parser = argparse.ArgumentParser(description='prepare data files for nematus to run.')
    parser.add_argument('--config', nargs=1, help='the config file, see nematus.ini as an example', required=True)
    args = parser.parse_args()        
    configfile = args.config[0]
    logger.info("config file: " + configfile)
    
    return {'configfile':configfile,}

def parse_config_var(config, var_name):
    if not 'PREPDATA' in config:
        logger.error("config file does not have section: PREPDATA. Exit.")
        sys.exit(1)
        
    if config['PREPDATA'][var_name]:
        var_val = config['PREPDATA'][var_name]
        logger.info("parse config: " + var_name + "=" + var_val)
        return var_val
    else:
        logger.error("parse config: no " + var_name + " configured. Exit.")
        sys.exit(1)


def parse_config(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)
    
    dataprep = parse_config_var(config, 'dataprep')
    outdir   = parse_config_var(config, 'outdir')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    return {'dataprep': dataprep,
            'outdir'  : outdir,}

def check_outputfiles(outputfiles):
    for key in outputfiles:
        fname=outputfiles[key]
        if not os.path.isfile(fname):
            return

    logger.info("the data files exists. Exit.")
    sys.exit()
    
########
######## Entry point
########

#
# dat-->src, com-->tgt
#

if __name__ == '__main__':
    args      = parse_args()
    config    = parse_config(args['configfile'])
    inputdir  = config['dataprep']
    outputdir = config['outdir']    

    ## input files
    srctokfile=os.path.join(inputdir, 'datstokenizer.pkl')
    tgttokfile=os.path.join(inputdir, 'comstokenizer.pkl')
    alldatafile=os.path.join(inputdir, 'alldata.pkl')
    
    ## output files
    outputfiles={
        'srcvocabfile':'vocab.src.json',
        'tgtvocabfile':'vocab.tgt.json',
        'srctrainfile':'train.src.txt',
        'tgttrainfile':'train.tgt.txt',
        'srcvalidfile':'valid.src.txt',
        'tgtvalidfile':'valid.tgt.txt',
        'srctestfile' : 'test.src.txt',
        'tgttestfile' : 'test.tgt.txt',}
    for key in outputfiles:
        outputfiles[key] = os.path.join(outputdir, outputfiles[key])
    
    check_outputfiles(outputfiles)
    
    ## loading files
    alldata = pickle.load(open(alldatafile, 'rb'))        
    srctok = pickle.load(open(srctokfile, 'rb'), encoding="UTF-8")
    index_word_src = {y:x for x,y in srctok.word_index.items()}
    tgttok = pickle.load(open(tgttokfile, 'rb'), encoding="UTF-8")
    index_word_tgt = {y:x for x,y in tgttok.word_index.items()}
    
    # generating vocab files
    logger.info("creating the vocab files...")
    vocabfiles(srctok, tgttok, srcvocabfile, tgtvocabfile)

    # generating training data files
    logger.info("creating the training data files...")
    getdata_from_alldata(alldata, 'dats_train_seqs', 'coms_train_seqs', index_word_src, index_word_tgt, srctrainfile, tgttrainfile)

    # generating valid data files
    # like the alpha version, for now, we use a subset from the training set as the valid set
    logger.info("creating the valid data files...")
    validfiles(alldata, index_word_src, index_word_tgt, srcvalidfile, tgtvalidfile)

    # generating test data files
    logger.info("creating the test data files...")
    getdata_from_alldata(alldata, 'dats_test_seqs', 'coms_test_seqs', index_word_src, index_word_tgt, srctestfile, tgttestfile)

    logger.info("Finished.")
