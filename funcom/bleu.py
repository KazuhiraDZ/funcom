import sys
import pickle
import argparse

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

from myutils import prep, drop, statusout, divideseqs, createbatchgen, seq2sent, index2word

def fil(com):
    ret = list()
    for w in com:
        if not '<' in w:
            ret.append(w)
    return ret

def bleu_so_far(refs, preds):
    Ba = corpus_bleu(refs, preds)
    B1 = corpus_bleu(refs, preds, weights=(1,0,0,0))
    B2 = corpus_bleu(refs, preds, weights=(0,1,0,0))
    B3 = corpus_bleu(refs, preds, weights=(0,0,1,0))
    B4 = corpus_bleu(refs, preds, weights=(0,0,0,1))

    Ba = round(Ba * 100, 2)
    B1 = round(B1 * 100, 2)
    B2 = round(B2 * 100, 2)
    B3 = round(B3 * 100, 2)
    B4 = round(B4 * 100, 2)

    ret = ''
    ret += ('Ba %s\n' % (Ba))
    ret += ('B1 %s\n' % (B1))
    ret += ('B2 %s\n' % (B2))
    ret += ('B3 %s\n' % (B3))
    ret += ('B4 %s\n' % (B4))
    
    return ret

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model-type', dest='modeltype', type=str, default='vanilla-lstm')
    parser.add_argument('--data-prep', dest='dataprep', type=str, default='../data/old')
    parser.add_argument('--outdir', dest='outdir', type=str, default='outdir')
    args = parser.parse_args()
    
    outdir = args.outdir
    dataprep = args.dataprep
    modeltype = args.modeltype

    sys.path.append(dataprep)
    import Tokenizer

    prep('preparing predictions list... ')
    preds = dict()
    predicts = open('%s/predict-%s.txt' % (outdir, modeltype), 'r')
    for c, line in enumerate(predicts):
        (fid, pred) = line.split('\t')
        fid = int(fid)
        pred = pred.split()
        pred = fil(pred)
        if(len(pred) == 0):
            continue
        preds[fid] = pred
    predicts.close()
    drop()

    refs = list()
    newpreds = list()
    d = 0
    targets = open('%s/reference.txt' % (outdir), 'r')
    for line in targets:
        (fid, com) = line.split('\t')
        fid = int(fid)
        com = com.replace('fonts', 'UNK')
        com = com.split()
        com = fil(com)

        if com[0] in ['see', 'non', 'begin', 'get', 'gets', 'set', 'sets']:
            continue
        
        try:
            newpreds.append(preds[fid])
        except KeyError as ex:
            continue
        
        refs.append([com])
        
        d += 0
        if(d > 0 and d % 10000 == 0):
            print('status at %s:' % (d))
            print(bleu_so_far(refs, preds[0:len(refs)]))
            
            print('sample output:')
            for i in range(d-10, d):
                print('%s' % preds[i])
                print('%s' % refs[i])
                print()
            print()

    print('final status')
    print(bleu_so_far(refs, newpreds))

