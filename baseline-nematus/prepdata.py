import time, logging, sys, os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import pickle
import json
import operator

import configparser, argparse
import re

def createvocab( trainset ):
    word_counts = dict()
    
    for fid, line in trainset:
        for word in line.split():
            if word == 'eos' or word == 'UNK':
                word = 'myword_' + word
    
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
    
    wcounts = list(word_counts.items())
    wcounts.sort(key=lambda x: x[1], reverse=True)
    
    vocab = ['eos', 'UNK']
    vocab.extend(wc[0] for wc in wcounts)
    
    return dict(list(
        zip(vocab, list(range(0, len(vocab))))
    ))


def vocabfile( trainset, outfile ):
    vocab = createvocab(trainset)
    
    out = open( outfile, 'w')
    json.dump(vocab, out, indent=2, ensure_ascii=False)
    out.close()


def createfile(dataset1, outfile1, dataset2, outfile2, maxlen_src, maxlen_tgt):
    logger.info("write to " + outfile1 + " and " + outfile2)
    
    with open(outfile1, mode='wt', encoding='utf-8') as outf1, open(outfile1+'.id', mode='wt', encoding='utf-8') as outfid1, open(outfile2, mode='wt', encoding='utf-8') as outf2, open(outfile2+'.id', mode='wt', encoding='utf-8') as outfid2:
        for (fid1, line1), (fid2, line2) in zip(dataset1, dataset2):
            if fid1 != fid2:
                logger.error("Error: " + outfile1 + " is not aligned with " + outfile2
                             + "\n fid1: " + str(fid1) + " fid2: " + str(fid2) + " are on the same line number.")
                sys.exit(1)
                
            line1words = line1.split()[:maxlen_src]
            line1 = ' '.join(line1words)    
            if line1 != '': # assume dataset1 is the source file
                outf1.write(line1+'\n')
                outfid1.write(str(fid1)+'\t'+line1+'\n')
                line2words = line2.split()[:maxlen_tgt]
                line2 = ' '.join(line2words)
                outf2.write(line2+'\n')
                outfid2.write(str(fid2)+'\t'+line2+'\n')


def loadfile(datafile):
    with open(datafile) as f:
        lines = f.read().splitlines()

    lines = [ line.split(',', maxsplit=1) for line in lines ]
    lines = [ (int(line[0].strip()), line[1].strip()) for line in lines ]
    return lines

            
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
    comtrain = parse_config_var(config, 'comtrain')
    comtest  = parse_config_var(config, 'comtest')
    comval   = parse_config_var(config, 'comval')
    dattrain = parse_config_var(config, 'dattrain')
    dattest  = parse_config_var(config, 'dattest')
    datval   = parse_config_var(config, 'datval')
    maxlen_tgt = int(parse_config_var(config, 'maxlen_tgt'))
    maxlen_src = int(parse_config_var(config, 'maxlen_src'))
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    return {'dataprep': dataprep,
            'outdir'  : outdir,
            'tgttrain': comtrain,
            'tgttest' : comtest,
            'tgtval'  : comval,
            'srctrain': dattrain,
            'srctest' : dattest,
            'srcval'  : datval,
            'maxlen_tgt' : maxlen_tgt,
            'maxlen_src' : maxlen_src,}

def check_outputfiles(outputfiles):
    for key in outputfiles:
        fname=outputfiles[key]
        if not os.path.isfile(fname):
            return

    logger.info("!!!!\n!!!!the data files exists. Exit.\n!!!!")
    sys.exit(0)
    
def check_inputfiles(inputfiles):
    for key in inputfiles:
        fname=inputfiles[key]
        if not os.path.isfile(fname):
            logger.error("!!!!\n!!!!missing input file: " + fname + ". Exit.\n!!!!")
            sys.exit(1)

def linecount(filename):
    num_lines = sum(1 for line in open(filename))
    logger.info(filename + ": " + str(num_lines) + "lines.")
    return num_lines

def sanity_check(outputfiles):
    lc1 = linecount(outputfiles['srcvocabfile'])
    lc2 = linecount(outputfiles['tgtvocabfile'])
    lc3 = linecount(outputfiles['srctrainfile'])
    lc4 = linecount(outputfiles['tgttrainfile'])
    if (lc3 != lc4):
        logger.error("\n\nThe src train file and the tgt train file have different line numbers!!!\n\n")
        
    lc5 = linecount(outputfiles['srcvalidfile'])
    lc6 = linecount(outputfiles['tgtvalidfile'])
    if (lc5 != lc6):
        logger.error("\n\nThe src valid file and the tgt valid file have different line numbers!!!\n\n")
    
    lc7 = linecount(outputfiles['srctestfile' ])
    lc8 = linecount(outputfiles['tgttestfile' ])
    if (lc7 != lc8):
        logger.error("\n\nThe src test file and the tgt test file have different line numbers!!!\n\n")
    

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
    maxlen_tgt = config['maxlen_tgt']
    maxlen_src = config['maxlen_src']

    ## input files
    inputfiles={
        'tgttrain': config['tgttrain'],
        'tgttest' : config['tgttest'],
        'tgtval'  : config['tgtval'],
        'srctrain': config['srctrain'],
        'srctest' : config['srctest'],
        'srcval'  : config['srcval'],}
    for key in inputfiles:
        inputfiles[key] = os.path.join(inputdir, inputfiles[key])

    check_inputfiles(inputfiles)
    
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
    
    # loading the input files
    logger.info("loading the train files...")
    datasets=dict()
    for key in inputfiles:
        datasets[key] = loadfile(inputfiles[key])
    
    # generating vocab files
    logger.info("creating the vocab files...")
    vocabfile(datasets['srctrain'], outputfiles['srcvocabfile'])
    vocabfile(datasets['tgttrain'], outputfiles['tgtvocabfile'])

    # generating training data files
    logger.info("creating the training data files...")
    createfile(datasets['srctrain'], outputfiles['srctrainfile'], datasets['tgttrain'], outputfiles['tgttrainfile'], maxlen_src, maxlen_tgt)

    # generating valid data files
    # like the alpha version, for now, we use a subset from the training set as the valid set
    logger.info("creating the valid data files...")
    createfile(datasets['srcval'], outputfiles['srcvalidfile'], datasets['tgtval'], outputfiles['tgtvalidfile'], maxlen_src, maxlen_tgt)

    # generating test data files
    logger.info("creating the test data files...")
    createfile(datasets['srctest'], outputfiles['srctestfile'], datasets['tgttest'], outputfiles['tgttestfile'], maxlen_src, maxlen_tgt)

    sanity_check(outputfiles)
    logger.info("Finished.")
