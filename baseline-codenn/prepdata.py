import time, logging, sys, os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import pickle
import json
import operator
import random
random.seed(1337)

import keras
# from keras.preprocessing.text import Tokenizer
import Tokenizer

import configparser, argparse
import re

def createvocab( origvocab ):
    vocab = dict()
    vocab['eos'] = 0
    vocab['UNK'] = 1
    vocab['<s>'] = 2
    vocab['</s>'] = 3
    
    offset = 4
    for word, count in origvocab.items():
        if word == 'eos' or word == 'UNK' or word == '<s>' or word == '</s>':
            continue
        
        vocab[word] = count + offset

    return vocab


def vocabfiles( srctok, tgttok, srcoutfile, tgtoutfile ):
    srcvocab = srctok.w2i
    tgtvocab = tgttok.w2i

    srcvocab = createvocab(srcvocab)
    tgtvocab = createvocab(tgtvocab)
    
    srcout = open( srcoutfile, 'w')
    json.dump(srcvocab, srcout, indent=2, ensure_ascii=False)
    srcout.close()    

    tgtout = open( tgtoutfile, 'w')
    json.dump(tgtvocab, tgtout, indent=2, ensure_ascii=False)
    tgtout.close()

def strip_newline(word_str):
    stripped_str = word_str
    stripped_str = re.sub(r"[\r\n]", "\\n", word_str)
    stripped_str = re.sub(r"\t", " ", stripped_str)
        
    return stripped_str.strip()
    

def print_seq_str(seq, index_word, index_type):
    print_str = ''
    for word in seq:
        if word == 0: # for the padding in the dat sequences
            break
        
        word_str = strip_newline(index_word[word])
        print_str += word_str + ' '
        
    return print_str


def getdata_from_alldata(alldata, field_src, field_tgt, index_word_src, index_word_tgt, srcoutfile, tgtoutfile):
    srclist = list()
    tgtlist = list()
    
    for fid in sorted(alldata[field_tgt].keys()):
        src = alldata[field_src][fid]
        src_str = strip_newline(src_str)
        srclist.append(src_str)

        tgt = alldata[field_tgt][fid]
        tgt_str = print_seq_str(tgt, index_word_tgt, field_tgt)
        tgtlist.append(tgt_str)

    logger.info("srclist: " + str(len(srclist)) + ", tgtlist: " + str(len(tgtlist)))
    
    with open(srcoutfile, mode='wt', encoding='utf-8') as outf:
        for line in srclist:
            outf.write(line+'\n')

    with open(tgtoutfile, mode='wt', encoding='utf-8') as outf:
        for line in tgtlist:
            outf.write(line+'\n')


def validfiles(alldata, field_src, field_tgt, index_word_src, index_word_tgt, srcoutfile, tgtoutfile):
    srclist = list()
    tgtlist = list()

    keys = sorted(alldata[field_tgt].keys())
    random.shuffle(keys)

    cnt = 0
    for fid in keys:
        cnt += 1
        src = alldata[field_src][fid]
        src_str = re.sub(r"\t", " ", src)
        srclist.append(src_str)

        tgt = alldata[field_tgt][fid]
        tgt_str = print_seq_str(tgt, index_word_tgt, 'tgt')
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
    # src_vocabsize = parse_config_var(config, 'vocabsize_src')
    # tgt_vocabsize = parse_config_var(config, 'vocabsize_tgt')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    return {'dataprep': dataprep,
            'outdir'  : outdir,}
            # 'vocabsize_src':src_vocabsize,
            # 'vocabsize_tgt':tgt_vocabsize,}

def check_outputfiles(outputfiles):
    for key in outputfiles:
        fname=outputfiles[key]
        if not os.path.isfile(fname):
            return

    logger.error("!!!!\n!!!!the data files exists. Exit.\n!!!!")
    sys.exit()

def check_two_files(file1, file2):
    num_lines1 = sum(1 for line in open(file1))
    num_lines2 = sum(1 for line in open(file2))
    if num_lines1 != num_lines2:
        logger.error("\n!!!!\n!!!!" + file1 + " has " + str(num_lines1) + " lines while " +
                     file2 + " has " + str(num_lines2) + " lines. Exit." +"\n!!!!\n")
        sys.exit()

    return

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

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

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
    srctok = pickle.load(open(srctokfile, 'rb'), encoding="UTF-8")
    index_word_src = srctok.i2w
    tgttok = pickle.load(open(tgttokfile, 'rb'), encoding="UTF-8")
    index_word_tgt = tgttok.i2w
    
    # if "\n" in index_word_src[208299]:
    #     logger.info("word has \\n: " +  str(srctok.word_index["else\n"]))
    # else:
    #     logger.info("word: " +  str(srctok.word_index["else"]))

    alldata = pickle.load(open(alldatafile, 'rb'))
    
    # generating vocab files
    logger.info("creating the vocab files...")
    vocabfiles(srctok, tgttok, outputfiles['srcvocabfile'], outputfiles['tgtvocabfile']) # , vocabsize_src, vocabsize_tgt

    # generating training data files
    logger.info("creating the training data files...")
    getdata_from_alldata(alldata, 'dats_raw', 'coms_train_seqs', index_word_src, index_word_tgt, outputfiles['srctrainfile'], outputfiles['tgttrainfile'])
    check_two_files(outputfiles['srctrainfile'], outputfiles['tgttrainfile'])

    # generating valid data files
    # like the alpha version, for now, we use a subset from the training set as the valid set
    logger.info("creating the valid data files...")
    validfiles(alldata, 'dats_raw', 'coms_train_seqs', index_word_src, index_word_tgt, outputfiles['srcvalidfile'], outputfiles['tgtvalidfile'])
    check_two_files(outputfiles['srcvalidfile'], outputfiles['tgtvalidfile'])

    # generating test data files
    logger.info("creating the test data files...")
    getdata_from_alldata(alldata, 'dats_raw', 'coms_test_seqs', index_word_src, index_word_tgt, outputfiles['srctestfile'], outputfiles['tgttestfile'])
    check_two_files(outputfiles['srctestfile'], outputfiles['tgttestfile'])

    logger.info("Finished.")
