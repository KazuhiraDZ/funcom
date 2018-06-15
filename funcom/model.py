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

def create_model(modeltype, config):
    mdl = None

    if modeltype == 'vanilla-lstm':
        mdl = VanillaLSTMModel(config)
    elif modeltype == 'vanilla-gru':
        mdl = VanillaGRUModel(config)
    elif modeltype == 'bidir-gru':
        mdl = BidirGRUModel(config)
    elif modeltype == 'attend-gru':
        mdl = AttentionGRUModel(config)
    elif modeltype == 'collin-1':
        mdl = Collin1Model(config)

    return mdl.create_model()
