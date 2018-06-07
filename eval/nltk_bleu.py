from nltk.translate.bleu_score import corpus_bleu
import argparse

def read_ref_file(fname):
    lines=list()
    with open(fname) as f:
        for line in f:
            toks=line.split()
            lines.append([toks])
        
    return lines

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

    refs=read_ref_file(ref_fname)
    preds=read_pred_file(pred_fname)

    print('reference file has ' + str(len(refs)) + ' lines')
    print('prediction file has ' + str(len(preds)) + ' lines')

    bleu=corpus_bleu(refs, preds, weights=(0.25, 0.25, 0.25, 0.25))
    bleu1=corpus_bleu(refs, preds, weights=(1, 0, 0, 0))
    bleu2=corpus_bleu(refs, preds, weights=(0, 1, 0, 0))
    bleu3=corpus_bleu(refs, preds, weights=(0, 0, 1, 0))
    bleu4=corpus_bleu(refs, preds, weights=(0, 0, 0, 1))
    print('BLEU: ' + str(bleu) + ' (' + str(bleu1) + ', ' + str(bleu2) + ', ' +  str(bleu3) + ', ' +  str(bleu4) + ')')
