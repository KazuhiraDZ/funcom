import os, sys, time, logging
import numpy as np


class Prediction:
    def __init__(self, search='greedy', search_params={}):
        """
        :param search: the name of search algorithm
        :param search_params: the dictionary that pass the parameter values to the search algorithm
        """
        self.logger = logging.getLogger(__name__ + ': Prediction')
        self.logger.info('Using ' + search + 'search for prediction')

        if search == 'greedy':
            self.search = self.greedy_search
        elif search == 'beam':
            self.search = self.beam_search

        self.params = search_params
        return

    def beam_search(self, predict, source_sequence, coms_vocabsize, max_caplen, sos, eos):
        k = self.params['beamwidth']
        
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
    def greedy_search(self, predict, source_sequence, coms_vocabsize, max_caplen, sos, eos):
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

