from nltk.translate.bleu_score import corpus_bleu
import argparse

def read_ref_file(fname):
    lines=list()
    with open(fname) as f:
        for line in f:
            toks=line.split()
            lines.append([toks])
        
    return lines

def read_pred_ref_file(predf, reff):
    preds=list()
    refs=list()

    with open(predf) as textfile1, open(reff) as textfile2: 
        for x, y in zip(textfile1, textfile2):
            x = x.strip()
            y = y.strip()
            
            toks=x.split()
            preds.append(toks)
            toks2=y.split()
            refs.append([toks2])

            
    return preds, refs

def read_pred_file(fname):
    lines=list()
    with open(fname) as f:
        for line in f:
            toks=line.split()
            lines.append(toks)
        
    return lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_file")
    parser.add_argument("predict_file")
    args = parser.parse_args()

    ref_fname = args.reference_file
    pred_fname = args.predict_file

    preds,refs=read_pred_ref_file(pred_fname, ref_fname)

    print('reference file has ' + str(len(refs)) + ' lines')
    print('prediction file has ' + str(len(preds)) + ' lines')

    bleu=corpus_bleu(refs, preds, weights=(0.25, 0.25, 0.25, 0.25))*100
    bleu1=corpus_bleu(refs, preds, weights=(1, 0, 0, 0))*100
    bleu2=corpus_bleu(refs, preds, weights=(0, 1, 0, 0))*100
    bleu3=corpus_bleu(refs, preds, weights=(0, 0, 1, 0))*100
    bleu4=corpus_bleu(refs, preds, weights=(0, 0, 0, 1))*100
    print('BLEU: ' + "{0:0.2f}".format(bleu) + ' (' + "{0:0.2f}".format(bleu1) + ', ' + "{0:0.2f}".format(bleu2) + ', ' +  "{0:0.2f}".format(bleu3) + ', ' +  "{0:0.2f}".format(bleu4) + ')')
