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
    ret += ('for %s functions\n' % (len(preds)))
    ret += ('Ba %s\n' % (Ba))
    ret += ('B1 %s\n' % (B1))
    ret += ('B2 %s\n' % (B2))
    ret += ('B3 %s\n' % (B3))
    ret += ('B4 %s\n' % (B4))
    
    return ret

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model-type-a', dest='modeltypeA', type=str, default='vanilla-lstm')
    parser.add_argument('--model-type-b', dest='modeltypeB', type=str, default='vanilla-lstm')
    parser.add_argument('--data-prep', dest='dataprep', type=str, default='../data/old')
    parser.add_argument('--outdir', dest='outdir', type=str, default='outdir')
    args = parser.parse_args()
    
    outdir = args.outdir
    dataprep = args.dataprep
    modeltypeA = args.modeltypeA
    modeltypeB = args.modeltypeB

    sys.path.append(dataprep)
    import Tokenizer

    prep('preparing predictions list... ')
    predsA = dict()
    predicts = open('%s/predict-%s.txt' % (outdir, modeltypeA), 'r')
    for c, line in enumerate(predicts):
        (fid, pred) = line.split('\t')
        fid = int(fid)
        pred = pred.split()
        pred = fil(pred)
        if(len(pred) == 0):
            continue
        predsA[fid] = pred
    predicts.close()
    drop()

    prep('preparing predictions list... ')
    predsB = dict()
    predicts = open('%s/predict-%s.txt' % (outdir, modeltypeB), 'r')
    for c, line in enumerate(predicts):
        (fid, pred) = line.split('\t')
        fid = int(fid)
        pred = pred.split()
        pred = fil(pred)
        if(len(pred) == 0):
            continue
        predsB[fid] = pred
    predicts.close()
    drop()

    #remfids = pickle.load(open('%s/remfids.pkl' % (outdir), 'rb'))
    #remfids = list()

    #remfidsf = open('%s/remfids.txt' % (outdir), 'r')

    #for line in remfidsf:
    #    remfids.append(int(line))

    refs = dict()
    newpreds = list()
    d = 0
    targets = open('%s/reference.txt' % (outdir), 'r')
    for line in targets:
        (fid, com) = line.split('\t')
        fid = int(fid)

        com = com.split()
        com = fil(com)
        
        refs[fid] = com

    betterA = list()
    betterB = list()
    ties = list()

    for fid in refs:
        ref = refs[fid]
        
        try:
            predA = predsA[fid]
            predB = predsB[fid]
        except KeyError as ex:
            continue
        
        bleuA = sentence_bleu([ref], predA, weights=(1,0,0,0))
        bleuB = sentence_bleu([ref], predB, weights=(1,0,0,0))
        
        if bleuA > bleuB:
            betterA.append(fid)
        elif bleuA < bleuB:
            betterB.append(fid)
        else:
            ties.append(fid)

    print(len(betterA))
    print(len(betterB))
    print(len(ties))

    betterAout = open('%s/betterA-%s.txt' % (outdir, modeltypeA), 'w')
    betterBout = open('%s/betterB-%s.txt' % (outdir, modeltypeB), 'w')

    for fid in betterA:
        betterAout.write('%s\t%s\t%s\t%s\n' % (fid, ' '.join(predsA[fid]), ' '.join(predsB[fid]), ' '.join(refs[fid])))

    for fid in betterB:
        betterBout.write('%s\t%s\t%s\t%s\n' % (fid, ' '.join(predsA[fid]), ' '.join(predsB[fid]), ' '.join(refs[fid])))

    betterAout.close()
    betterBout.close()

    #print('final status')
    #print(bleu_so_far(refs, newpreds))

