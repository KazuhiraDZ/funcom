from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, Bidirectional
from keras.optimizers import RMSprop
import keras
import tensorflow as tf

class VanillaGRUModel:
    def __init__(self, config):
        self.datvocabsize = config['datvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.datlen = config['datlen']
        self.comlen = config['comlen']
        self.multigpu = config['multigpu']

    def create_model(self):
        dat_input = Input(shape=(self.datlen,))
        com_input = Input(shape=(self.comlen,))

        xd = Embedding(output_dim=100, input_dim=self.datvocabsize, mask_zero=True)(dat_input)
        ld = GRU(256, return_state=True, activation='tanh')
        ldout, state_h = ld(xd)

        xc = Embedding(output_dim=100, input_dim=self.comvocabsize, mask_zero=True)(com_input)
        lc = GRU(256)
        lcout = lc(xc, initial_state=state_h)

        out = Dense(self.comvocabsize, activation='softmax')(lcout)

        model = Model(inputs=[dat_input, com_input], outputs=out)
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.0001, clipnorm=1.), metrics=['accuracy'])
        return model
