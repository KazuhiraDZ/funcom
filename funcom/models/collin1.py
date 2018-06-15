from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, Bidirectional, CuDNNGRU, RepeatVector, Permute, TimeDistributed, dot, SimpleRNN
from keras.optimizers import RMSprop, Adamax
import keras
import keras.utils
import tensorflow as tf

class Collin1Model:
    def __init__(self, config):
        self.datvocabsize = config['datvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.datlen = config['datlen']
        self.comlen = config['comlen']
        self.multigpu = config['multigpu']
        self.batch_size = config['batch_size']
        
        self.embdims = 100
        self.recdims = 256
    
    def create_model(self):
        
        dat_input = Input(shape=(self.datlen,))
        com_input = Input(shape=(self.comlen,))
        
        ee = Embedding(output_dim=self.embdims, input_dim=self.datvocabsize, mask_zero=False)(dat_input)
        
        #enc = GRU(self.recdims, return_state=True, return_sequences=True, activation='tanh', unroll=True)
        enc = SimpleRNN(self.recdims, return_state=True, return_sequences=True, unroll=True)
        encout, state_h = enc(ee)
        
        de = Embedding(output_dim=self.embdims, input_dim=self.comvocabsize, mask_zero=False)(com_input)
        #dec = GRU(self.recdims, return_sequences=True, activation='tanh', unroll=True)
        dec = CuDNNGRU(self.recdims, return_sequences=True)
        decout = dec(de, initial_state=state_h)

        attn = dot([decout, encout], axes=[2, 2])
        attn = Activation('softmax')(attn)

        context = dot([attn, encout], axes=[2,1])
        context = concatenate([context, decout])
        
        out = TimeDistributed(Dense(self.recdims, activation="tanh"))(context)

        out = Flatten()(out)
        out = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[dat_input, com_input], outputs=out)
        
        if(self.multigpu):
            model = keras.utils.multi_gpu_model(model, gpus=2)
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
