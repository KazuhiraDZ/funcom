from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten
from keras.optimizers import RMSprop
import keras
import tensorflow as tf

def fun_com_model(coms_vocabsize=2187, dats_vocabsize=1525916, dats_emb=None, lang_dim=100,
            max_comlen=50, max_datlen=50, clipnorm=1):

    lang_input = Input(shape=(1,))
    dat_input = Input(shape=(max_datlen,))
    seq_input = Input(shape=(max_comlen,))
    vhist_input = Input(shape=(coms_vocabsize,))

    ### comment text input layers
    
    # could load weights to start from a pre-trained embedding
    with tf.device('/cpu:0'):
        x = Embedding(output_dim=100, input_dim=coms_vocabsize, input_length=1, embeddings_initializer="glorot_uniform")(lang_input)

    lang_embed = Reshape((lang_dim,))(x)
    lang_embed = concatenate([lang_embed, seq_input])
    lang_embed = Dense(lang_dim)(lang_embed)
    lang_embed = Dropout(0.25)(lang_embed)

    ### function contents text input layers
    
    with tf.device('/cpu:0'):
        e = Embedding(output_dim=100, input_dim=dats_vocabsize, input_length=max_datlen, embeddings_initializer="glorot_uniform", weights=[dats_emb], trainable=False)(dat_input)
    
    filters = 100
    kernel_size = 3
    dat_conv = Conv1D(filters, kernel_size, activation='relu', strides=1)(e)
    dat_conv = MaxPooling1D()(dat_conv)
    dat_conv = Flatten()(dat_conv)
    dat_conv_dim = int(filters * ((max_datlen/2)-1))

    ### merged layers

    merge_layer = concatenate([dat_conv, lang_embed, vhist_input])
    merge_layer = Reshape((1, lang_dim+dat_conv_dim+coms_vocabsize))(merge_layer)

    gru_1 = GRU(dat_conv_dim)(merge_layer)
    gru_1 = Dropout(0.25)(gru_1)
    gru_1 = Dense(dat_conv_dim)(gru_1)
    gru_1 = BatchNormalization()(gru_1)
    gru_1 = Activation('softmax')(gru_1)

    attention_1 = multiply([dat_conv, gru_1])
    attention_1 = concatenate([attention_1, lang_embed, vhist_input])
    attention_1 = Reshape((1, lang_dim+dat_conv_dim+coms_vocabsize))(attention_1)
    gru_2 = GRU(1024)(attention_1)
    gru_2 = Dropout(0.25)(gru_2)
    gru_2 = Dense(coms_vocabsize)(gru_2)
    gru_2 = BatchNormalization()(gru_2)
    out = Activation('softmax')(gru_2)

    model = Model(inputs=[dat_input, lang_input, seq_input, vhist_input], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.0001, clipnorm=1.))
    return model
