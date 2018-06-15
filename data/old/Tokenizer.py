from collections import Counter, defaultdict
from gensim.models import Word2Vec
import pickle
import numpy as np

# Get trained word vectors from the trained W2V model
# and create a tokenizer
# Basic Functionality:
#		word_counts: count for each word in vocab
# 		w2i: dictionary word -> index map
#		i2w: dictionary index -> word map
#		vectors: dictionary word -> word vector
#		oov_index: this will be set to always be the last index in the vocab.
#					If you change vocab size in text_to_sequences oov_index also changes
#		oov_vector: set in makew2v.py, default is avg of 50 least commong word vectors
#		oov_token: set to 'UNK' but can be anything

class Tokenizer(object):
	def __init__(self, oov_token='UNK'):
		self.word_counts = Counter()
		self.w2i = {}
		self.i2w = {}
		self.vectors = {}
		self.oov_index = None
		self.oov_vector = None
		self.oov_token = oov_token
		self.vocab_size = None

	def save(self, path):
		pickle.dump(self, open(path, 'wb'))

	def load_from_w2v(self, path=None, model=None):
		if path is not None:
			l = Word2Vec.load(path)

		if model is not None:
			l = model

		for w in l.wv.vocab.keys():
			self.word_counts[w] = l.wv.vocab[w].count
			self.vectors[w] = l.wv[w]

		# 0 is a reserved index
		for count, w in enumerate(self.word_counts.most_common()):
			self.w2i[w[0]] = count+1
			self.i2w[count+1] = w[0]


		self.w2i['<mask>'] = 0
		self.i2w[0] = '<mask>'
		self.vectors['<mask>'] = np.zeros(len(l.wv[list(l.wv.vocab.keys())[0]]))
		self.oov_index = len(list(self.w2i.keys()))
		self.vocab_size = self.oov_index

	def texts_to_sequences(self, texts, maxlen=None, vocab_size=None):
		all_seq = []
		tot = len(texts)
		count = 0
		allowed = None
		
		if vocab_size is not None:
			# we do vocab_size-1 so the index is correct (if vocab size is 50, index 49 will be the last one 0-49)
			self.vocab_size = vocab_size
			self.oov_index = vocab_size-1
			allowed = defaultdict(bool)
			for word, count in list(self.word_counts.most_common())[:vocab_size-1]:
				allowed[word] = True

		for text in texts:
			text = text.split()
			seq = []

			for w in text:
				if maxlen is not None:
					if len(seq) == maxlen:
						break

				# lets us limit vocab size quickly without removing words
				if allowed is not None:
					if allowed[w]:
						seq.append(self.w2i[w])
					else:
						seq.append(self.oov_index)
					continue

				try:
					seq.append(self.w2i[w])
				except:
					seq.append(self.oov_index)

			
			all_seq.append(np.array(seq))

		return all_seq

	def get_vectors(self):
		vec_list = []
		vec_list.append(self.vectors['<mask>'])
		# -1 for mask vector and -1 for unk vector
		for word, count in self.word_counts.most_common()[:self.vocab_size-2]:
			vec_list.append(np.array(self.vectors[word]))

		vec_list.append(np.array(self.oov_vector))

		return np.array(vec_list)