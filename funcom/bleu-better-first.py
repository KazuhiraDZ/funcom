import sys
import pickle
import argparse

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

from myutils import prep, drop, statusout, divideseqs, createbatchgen, seq2sent, index2word

import math
import nltk

# following two functions implement the LSS semantic similarity metric:

def compare_words( word_a, word_b, wup=False ):
        synsets_a = nltk.corpus.wordnet.synsets(word_a)
        synsets_b = nltk.corpus.wordnet.synsets(word_b)
        best = 0.0

        for a in synsets_a:
                for b in synsets_b:

                        if wup:
                                sim = a.wup_similarity(b)
                        else:
                                sim = a.path_similarity(b)

                        try:
                            best = max( sim, best )
                        except TypeError as ex:
                            pass

        return best

def titlesim( tok_a, tok_b ):
        words = [ i for i in set(tok_a+tok_b) if i not in nltk.corpus.stopwords.words('english') ]

        vector_a = { i: {'count': tok_a.count(i), 'weight': 0} for i in words }
        vector_b = { i: {'count': tok_b.count(i), 'weight': 0} for i in words }

        # generate weighted vectors
        for i in vector_a:
                for j in vector_a:
                        vector_a[i]['weight'] += vector_a[j]['count'] * compare_words( i, j )

        for i in vector_b:
                for j in vector_b:
                        vector_b[i]['weight'] += vector_b[j]['count'] * compare_words( i, j )

        # generate dot products
        dot = dot_a = dot_b = 0
        for i in words:
                dot = dot + (vector_a[i]['weight']*vector_b[i]['weight'])
                dot_a = dot_a + pow( vector_a[i]['weight'], 2 )
                dot_b = dot_b + pow( vector_b[i]['weight'], 2 )

        try:
            cos = dot / (math.sqrt(dot_a) * math.sqrt(dot_b))
        except Exception as ex:
            cos = 0.0

        return cos

### end LSS


def fil(com, revstem):
    ret = list()
    for w in com:
        if not '<' in w:
            try:
                ret.append(revstem[w])
            except KeyError as ex:
                ret.append(w)
    ret = ret[0]
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

    revstem = dict()
    revstemwords = open('%s/revstemcoms.txt' % (dataprep), 'r')
    for line in revstemwords:
        (word, stem) = line.split(',')
        stem = stem.rstrip()
        try:
            revstem[stem]
        except KeyError as ex: # use first occurrence
            revstem[stem] = word

    prep('preparing predictions list... ')
    predsA = dict()
    predicts = open('%s/predict-%s.txt' % (outdir, modeltypeA), 'r')
    for c, line in enumerate(predicts):
        (fid, pred) = line.split('\t')
        fid = int(fid)
        pred = pred.split()
        pred = fil(pred, revstem)
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
        pred = fil(pred, revstem)
        if(len(pred) == 0):
            continue
        predsB[fid] = pred
    predicts.close()
    drop()

    refs = dict()
    newpreds = list()
    d = 0
    targets = open('%s/reference.txt' % (outdir), 'r')
    for line in targets:
        (fid, com) = line.split('\t')
        fid = int(fid)

        com = com.split()
        com = fil(com, revstem)
        
        refs[fid] = com

    BbetterA = list()
    BbetterB = list()
    Bties = list()
    LbetterA = list()
    LbetterB = list()
    Lties = list()
    bleusA = dict()
    bleusB = dict()
    lssesA = dict()
    lssesB = dict()

    for (c, fid) in enumerate(refs):
        ref = refs[fid]
        
        try:
            predA = predsA[fid]
            predB = predsB[fid]
        except KeyError as ex:
            continue
        
        bleuA = sentence_bleu([ref], predA, weights=(1,0,0,0))
        bleuB = sentence_bleu([ref], predB, weights=(1,0,0,0))
        
        lssA = titlesim(predA, ref)
        lssB = titlesim(predB, ref)
        
        bleusA[fid] = bleuA
        bleusB[fid] = bleuB
        
        lssesA[fid] = lssA
        lssesB[fid] = lssB

        if bleuA > bleuB:
            BbetterA.append(fid)
        elif bleuA < bleuB:
            BbetterB.append(fid)
        else:
            Bties.append(fid)
            
        if lssA > lssB:
            LbetterA.append(fid)
        elif lssA < lssB:
            LbetterB.append(fid)
        else:
            Lties.append(fid)
            
        if(c > 0 and c % 100 == 0):
            print(c)
            
            print("BLEU:")
            print(len(BbetterA))
            print(len(BbetterB))
            print(len(Bties))

            print("LSS:")
            print(len(LbetterA))
            print(len(LbetterB))
            print(len(Lties))
            
            print()
            #break

    print("BLEU:")
    print(len(BbetterA))
    print(len(BbetterB))
    print(len(Bties))

    print("LSS:")
    print(len(LbetterA))
    print(len(LbetterB))
    print(len(Lties))

    betterAout = open('%s/BfbetterA-%s.txt' % (outdir, modeltypeA), 'w')
    betterBout = open('%s/BfbetterB-%s.txt' % (outdir, modeltypeB), 'w')

    for fid in BbetterA:
        betterAout.write('%s\t%s (%s)\t%s (%s)\t%s\n' % (fid, ' '.join(predsA[fid]), bleusA[fid],  ' '.join(predsB[fid]), bleusB[fid], ' '.join(refs[fid])))

    for fid in BbetterB:
        betterBout.write('%s\t%s (%s)\t%s (%s)\t%s\n' % (fid, ' '.join(predsA[fid]), bleusA[fid],  ' '.join(predsB[fid]), bleusB[fid], ' '.join(refs[fid])))

    betterAout.close()
    betterBout.close()

    betterAout = open('%s/LfbetterA-%s.txt' % (outdir, modeltypeA), 'w')
    betterBout = open('%s/LfbetterB-%s.txt' % (outdir, modeltypeB), 'w')

    for fid in LbetterA:
        betterAout.write('%s\t%s (%s)\t%s (%s)\t%s\n' % (fid, ' '.join(predsA[fid]), lssesA[fid],  ' '.join(predsB[fid]), lssesB[fid], ' '.join(refs[fid])))

    for fid in LbetterB:
        betterBout.write('%s\t%s (%s)\t%s (%s)\t%s\n' % (fid, ' '.join(predsA[fid]), lssesA[fid],  ' '.join(predsB[fid]), lssesB[fid], ' '.join(refs[fid])))

    betterAout.close()
    betterBout.close()

    #print('final status')
    #print(bleu_so_far(refs, newpreds))

