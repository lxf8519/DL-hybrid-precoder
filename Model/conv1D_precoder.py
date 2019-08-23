# Paper: Deep Learning for Direct Hybrid Precoding in Millimeter Wave Massive MIMO Systems  https://arxiv.org/abs/1905.13212

from keras.layers import Input, Dense, Conv1D, UpSampling1D, Activation, Flatten, Dropout
from keras.layers import Reshape
from keras.models import Model
from Complex_layers.conv import ComplexConv1D
from Model.network_base import NetworkBase
from keras.layers.core import Permute
from Complex_layers.bn import ComplexBatchNormalization as ComplexBN
from keras.layers import LeakyReLU
from keras import optimizers
from keras.optimizers import Adam
from keras.layers import BatchNormalization as BN, Concatenate
from keras.initializers import Constant
from keras import backend as K

class ConvPrecoder(NetworkBase):

    def __init__(self, len_in, dims, num_classes):

        N_BS = dims[0]
        N_MS = dims[1]
        M_BS = dims[2]
        M_MS = dims[3]

        input  = Input(shape=(len_in,2), name='EncoderIn')
        encoded = input

        encoded = (ComplexConv1D(M_BS, N_BS, strides=N_BS, padding='valid',
                    kernel_initializer='complex_independent',name='Conv_P'))(encoded)

        encoded = Permute([2, 1])(encoded)
        encoded = Reshape((2, N_MS*M_BS), name='Y_1_transpose')(encoded)
        encoded = Permute([2, 1])(encoded)
        encoded = ComplexConv1D(M_MS, N_MS, strides=N_MS, padding='valid',
                    kernel_initializer='complex_independent', name='Conv_Qconj')(encoded)



        encoded=Flatten()(encoded)

        dense_layer1=Dense(N_MS*N_BS,activation='sigmoid',bias_initializer=Constant(value=-5), name='FC_layer1')
        encoded = dense_layer1(encoded)

        drop_layer1 = Dropout(.05)
        encoded = drop_layer1(encoded)

        dense_layer2 = Dense(N_MS * N_BS, activation='sigmoid',bias_initializer=Constant(value=-5), name='FC_layer2')
        encoded = dense_layer2(encoded)

        drop_layer2 = Dropout(.05)
        encoded = drop_layer2(encoded)

        dense_layer_task1 = Dense(num_classes, bias_initializer=Constant(value=-1), name='FC_layer3')
        output1 = dense_layer_task1(encoded)

        dense_layer_task2 = Dense(num_classes, bias_initializer=Constant(value=-1), name='FC_layer4')
        output2 = dense_layer_task2(encoded)

        output=Concatenate(axis=1)([output1, output2])
        output=Activation('sigmoid')(output)

        self.fcnet = Model(inputs=input, outputs=output)

        def cat_loss(y_true, y_pred):
           loss =K.mean(K.binary_crossentropy(y_true[:, 0:num_classes-1], y_pred[:, 0:num_classes-1]),axis=-1)+K.mean(K.binary_crossentropy(y_true[:, num_classes:2*num_classes-1], y_pred[:, num_classes:2*num_classes-1]),axis=-1)


           return loss

        adam = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.fcnet.compile(optimizer=adam, loss=cat_loss)

    def compile(self,optimizer,loss):
        self.fcnet.compile(optimizer, loss)
