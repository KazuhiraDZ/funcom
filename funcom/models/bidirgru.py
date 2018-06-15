from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, Bidirectional, CuDNNGRU
from keras.optimizers import RMSprop
import keras
import tensorflow as tf

class BidirGRUModel:
    def __init__(self, datvocabsize, comvocabsize, datlen, comlen):
        self.datvocabsize = datvocabsize
        self.comvocabsize = comvocabsize
        self.datlen = datlen
        self.comlen = comlen
        
        self.embdims = 256
        self.recdims = 256
    
    def create_model(self):
        dat_input = Input(shape=(self.datlen,))
        com_input = Input(shape=(self.comlen,))

        xd = Embedding(output_dim=self.embdims, input_dim=self.datvocabsize, mask_zero=True, trainable=True)(dat_input)
        ld = Bidirectional(GRU(self.recdims, return_state=True))
        ldout, forward_h, backward_h = ld(xd)
        
        #ldf = GRU(self.recdims, return_state=True)
        #ldout, forward_h = ldf(xd)
        
        #ldb = GRU(self.recdims, return_state=True, go_backwards=True)
        #ldout, backward_h = ldb(xd)
        
        state_h = concatenate([forward_h, backward_h])

        xc = Embedding(output_dim=self.embdims, input_dim=self.comvocabsize, mask_zero=True, trainable=True)(com_input)
        lc = GRU(self.recdims * 2)
        lcout = lc(xc, initial_state=state_h)

        out = Dense(self.comvocabsize, activation='softmax')(lcout)

        model = Model(inputs=[dat_input, com_input], outputs=out)
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.0001, clipnorm=1.), metrics=['accuracy'])
        return model
