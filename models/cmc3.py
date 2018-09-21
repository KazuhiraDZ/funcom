from keras.models import Model
from keras.layers import Input, Maximum, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, MaxPooling2D, Conv1D, Conv2D, Flatten, Bidirectional, CuDNNGRU, RepeatVector, Permute, TimeDistributed, dot
from keras.backend import tile, repeat, repeat_elements
from keras.optimizers import RMSprop, Adamax
import keras
import keras.utils
import tensorflow as tf

class Cmc3Model:
    def __init__(self, config):
        
        # data length in dataset is 20+ functions per file, but we can elect to reduce
        # that length here, since myutils reads this length when creating the batches
        config['sdatlen'] = 10
        
        config['tdatlen'] = 50
        
        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.smlvocabsize = config['smlvocabsize']
        self.tdatlen = config['tdatlen']
        self.sdatlen = config['sdatlen']
        self.comlen = config['comlen']
        self.smllen = config['smllen']

        self.config['num_input'] = 4
        self.config['num_output'] = 1

        self.embdims = 100
        self.smldims = 10
        self.recdims = 256
        self.tdddims = 256
        
        self.filters = 32
        self.kern = (1,5)
        self.pool = (1,2)

    def create_model(self):
        
        tdat_input = Input(shape=(self.tdatlen,))
        #sdat_input = Input(shape=(self.sdatlen, self.tdatlen, 1))
        sdat_input = Input(shape=(self.sdatlen, self.tdatlen))
        com_input = Input(shape=(self.comlen,))
        sml_input = Input(shape=(self.smllen,))
        
        tdel = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)
        tde = tdel(tdat_input)
        #sde = Embedding(output_dim=self.embdims, input_dim=self.smlvocabsize, mask_zero=False)(sdat_input)
        se = Embedding(output_dim=self.smldims, input_dim=self.smlvocabsize, mask_zero=False)(sml_input)

        #se_emb = Conv1D(10, 3)(se)
        #se_emb = MaxPooling1D()(se_emb)
        #se_enc = Flatten()
        #seout = se_enc(se_emb)
        se_enc = CuDNNGRU(self.recdims, return_state=True, return_sequences=True)
        seout, state_sml = se_enc(se)

        tenc = CuDNNGRU(self.recdims, return_state=True, return_sequences=True)
        tencout, tstate_h = tenc(tde, initial_state=state_sml)
        
        #senc = CuDNNGRU(self.recdims, return_state=True, return_sequences=True)
        #sencout, sstate_h = senc(sde, initial_state=tstate_h)
        
        de = Embedding(output_dim=self.embdims, input_dim=self.comvocabsize, mask_zero=False)(com_input)
        #dec = GRU(self.recdims, return_sequences=True, activation='tanh', unroll=True)
        dec = CuDNNGRU(self.recdims, return_sequences=True)
        decout = dec(de, initial_state=tstate_h)

        tattn = dot([decout, tencout], axes=[2, 2])
        tattn = Activation('softmax')(tattn)

        #sattn = dot([decout, sencout], axes=[2, 2])
        #sattn = Activation('softmax')(sattn)

        ast_attn = dot([decout, seout], axes=[2, 2])
        ast_attn = Activation('softmax')(ast_attn)

        tcontext = dot([tattn, tencout], axes=[2, 1])
        #scontext = dot([sattn, sencout], axes=[2, 1])
        ast_context = dot([ast_attn, seout], axes=[2, 1])

        semb = TimeDistributed(tdel)
        sde = semb(sdat_input)
        
        senc = TimeDistributed(CuDNNGRU(int(self.recdims/2)))
        senc = senc(sde)
        
        scontext = Flatten()(senc)

        #sdatconv1 = Conv2D(self.filters, self.kern, activation='relu')
        #sdatconv1 = sdatconv1(sdat_input)
        #sdatconv1 = MaxPooling2D(pool_size=self.pool)(sdatconv1)

        #sdatconv2 = Conv2D(self.filters*2, self.kern, activation='relu')
        #sdatconv2 = sdatconv2(sdatconv1)
        #sdatconv2 = MaxPooling2D(pool_size=self.kern)(sdatconv2)

        #scontext = Flatten()(sdatconv1)
        #scontext = Dense(self.tdddims)(scontext)
        #out2 = Dense(self.comvocabsize, activation="softmax")(scontext)
        scontext = RepeatVector(self.comlen)(scontext)

        #senc = CuDNNGRU(self.recdims, return_sequences=True)
        #sencout = senc(sdatconv1, return_sequences=True)

        #sattn = dot([decout, sencout], axes=[2, 2])
        #sattn = Activation('softmax')(sattn)
        
        #scontext = dot([sattn, sencout], axes=[2, 1])

        #seout = RepeatVector(self.comlen)(seout)

        context = concatenate([scontext, tcontext, decout, ast_context])

        out = TimeDistributed(Dense(self.tdddims, activation="relu"))(context)

        out = Flatten()(out)
        #out = concatenate([seout, out])
        #out = Dense(2048, activation='relu')(out)
        out1 = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[tdat_input, sdat_input, com_input, sml_input], outputs=out1)
        #model = Model(inputs=[tdat_input, sdat_input, com_input, sml_input], outputs=[out1, out2])
        
        if self.config['multigpu']:
            model = keras.utils.multi_gpu_model(model, gpus=2)

        model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
        return self.config, model
