from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten
from keras.optimizers import RMSprop
import keras
import tensorflow as tf

from models.vanillalstm import VanillaLSTMModel
from models.vanillagru import VanillaGRUModel
from models.bidirgru import BidirGRUModel
from models.attendgru import AttentionGRUModel
from models.collin1 import Collin1Model

def create_model(modeltype, datvocabsize, comvocabsize, datlen=100, comlen=13, multigpu=False):
    mdl = None

    if modeltype == 'vanilla-lstm':
        mdl = VanillaLSTMModel(datvocabsize, comvocabsize, datlen, comlen)
    elif modeltype == 'vanilla-gru':
        mdl = VanillaGRUModel(datvocabsize, comvocabsize, datlen, comlen)
    elif modeltype == 'bidir-gru':
        mdl = BidirGRUModel(datvocabsize, comvocabsize, datlen, comlen)
    elif modeltype == 'attend-gru':
        mdl = AttentionGRUModel(datvocabsize, comvocabsize, datlen, comlen, multigpu)
    elif modeltype == 'collin-1':
        mdl = Collin1Model(datvocabsize, comvocabsize, datlen, comlen, multigpu)

    return mdl.create_model()
