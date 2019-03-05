from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten
from keras.optimizers import RMSprop
import keras
import tensorflow as tf

from models.attendgru import AttentionGRUModel
from models.ast_attendgru_xtra import AstAttentionGRUModel as xtra
from models.transformer import TransformerModel
from models.cmc5 import Cmc5Model as cmc5
from models.cmc6 import Cmc6Model as cmc6
from models.cmc7 import Cmc7Model as cmc7
from models.cmc8 import Cmc8Model as cmc8

def create_model(modeltype, config):
    mdl = None

    if modeltype == 'attendgru':
    	# base attention GRU model based on Nematus architecture
        mdl = AttentionGRUModel(config)
    elif modeltype == 'ast-attendgru':
    	# attention GRU model with added AST information from srcml. 
        mdl = xtra(config)
    elif modeltype == 'cmc5':
        # sandbox model to try things
        mdl = cmc5(config)
    elif modeltype == 'cmc6':
        # sandbox model to try things
        mdl = cmc6(config)
    elif modeltype == 'cmc7':
        # sandbox model to try things
        mdl = cmc7(config)
    elif modeltype == 'cmc8':
        # sandbox model to try things
        mdl = cmc8(config)
    elif modeltype == 'transformer':
        # sandbox model to try things
        mdl = TransformerModel(config)
    else:
        print("{} is not a valid model type".format(modeltype))
        exit(1)
        
    return mdl.create_model()
