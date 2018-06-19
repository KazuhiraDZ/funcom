import time, logging, sys, os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import pickle
import json
import operator

import configparser, argparse
import re

def strip_newline(word_str):
    stripped_str = word_str
    stripped_str = re.sub(r"[\r\n]", "\\\\<nl>", word_str)
    stripped_str = re.sub(r"\t", " ", stripped_str)

    return stripped_str.strip()


def createfile_src(inputfile, rawdataset, outfile):
    logger.info("loading: " + inputfile)
    dataset = loadfile(inputfile)
    logger.info("write to " + outfile)

    with open(outfile, mode='wt', encoding='utf-8') as outf, open(outfile+'.id', mode='wt', encoding='utf-8') as outfid:
        for fid, line in dataset:
            rawcode = rawdataset[fid]
            src = strip_newline(rawcode)
            outf.write(src+'\n')
            outfid.write(str(fid)+'\t'+src+'\n')


def createfile(inputfile, outfile):
    logger.info("laoding: " + inputfile)
    dataset = loadfile(inputfile)
    
    logger.info("write to " + outfile)

    with open(outfile, mode='wt', encoding='utf-8') as outf, open(outfile+'.id', mode='wt', encoding='utf-8') as outfid:
        for fid, line in dataset:
            outf.write(line+'\n')
            outfid.write(str(fid)+'\t'+line+'\n')


def loadfile(datafile):
    if datafile.endswith('.pkl'):
        return pickle.load(open(datafile, 'rb'), encoding="UTF-8")
    else:
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

    rawdats  = parse_config_var(config, 'rawdatspkl')
        
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
            'rawdats' : rawdats,}

def check_outputfiles(outputfiles):
    for key in outputfiles:
        fname=outputfiles[key]
        if not os.path.isfile(fname):
            return

    logger.info("!!!!\n!!!!the data files exists. Exit.\n!!!!")
    sys.exit()

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

    ## input files
    inputfiles={
        'tgttrain': config['tgttrain'],
        'tgttest' : config['tgttest'],
        'tgtval'  : config['tgtval'],
        'srctrain': config['srctrain'],
        'srctest' : config['srctest'],
        'srcval'  : config['srcval'],
        'rawdats' : config['rawdats'],}
    for key in inputfiles:
        inputfiles[key] = os.path.join(inputdir, inputfiles[key])

    check_inputfiles(inputfiles)

    ## output files
    outputfiles={
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
    logger.info("loading the raw data file: " + inputfiles['rawdats'])
    rawdataset = loadfile(inputfiles['rawdats'])
    logger.info("finished loading the raw data file.")
    
    # generating vocab files
    logger.info("skipping the vocab files because codenn will create them later ...")
    
    # generating training data files
    logger.info("creating the training data files...")
    createfile_src(inputfiles['srctrain'], rawdataset, outputfiles['srctrainfile'])
    createfile(inputfiles['tgttrain'], outputfiles['tgttrainfile'])

    # generating valid data files
    logger.info("creating the valid data files...")
    createfile_src(inputfiles['srcval'], rawdataset, outputfiles['srcvalidfile'])
    createfile(inputfiles['tgtval'], outputfiles['tgtvalidfile'])

    # generating test data files
    logger.info("creating the test data files...")
    createfile_src(inputfiles['srctest'], rawdataset, outputfiles['srctestfile'])
    createfile(inputfiles['tgttest'], outputfiles['tgttestfile'])


    sanity_check(outputfiles)
    logger.info("Finished.")
