# coding=gbk
from keras.layers import Embedding, Dense, Input, SimpleRNN, RNN
from keras.models import Model
from keras.layers import Layer
from keras import activations
import keras.backend as K


# Keras官方实现好了RNN，我们可以直接调用，学习时我们可以重复造轮子，自己实现一个简单的RNN层

#-----------------------实现简单的rnn_cell和rnn_layer-----------------------------
class My_SimpleRNN_Cell(Layer):
    def __init__(self, 
                 units, 
                 activation='tanh', 
                 use_bias=True, 
                 **kwargs):
        super().__init__(**kwargs)
        self.state_size = units
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        super().build(input_shape)
        self.u = self.add_weight(name='U', shape=(input_shape[-1], self.units))
        self.w = self.add_weight(name='W', shape=(self.units, self.units))
        self.v = self.add_weight(name='V', shape=(self.units, self.units))
        if self.use_bias:
            self.bu = self.add_weight(name='bu', shape=(self.units,))
            self.bw = self.add_weight(name='bw', shape=(self.units,))
            self.bv = self.add_weight(name='bv', shape=(self.units,))

    def call(self, inputs, states):
        x = K.dot(inputs, self.u)
        s = K.dot(states[0], self.w)
        if self.use_bias:
            x = K.bias_add(x, self.bu)
            s = K.bias_add(s, self.bw)
        st = self.activation(x + s)
        o = K.dot(st, self.v)
        if self.use_bias:
            o = K.bias_add(o, self.bv)
        o = self.activation(o)
        return o, [st]


class My_SimpleRNN_Layer(Layer):
    def __init__(self, 
                 units,
                 activation = 'tanh',
                 use_bias = True,
                 return_sequences = False,
                 **kwargs):
        super(My_SimpleRNN_Layer, self).__init__(**kwargs)
        self.units = units
        self.cell = My_SimpleRNN_Cell(units, activation, use_bias)
        self.layer = RNN(self.cell, return_sequences=return_sequences)

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs):
        x = self.layer(inputs)
        return x
#-----------------------------------------------------------------------------


#---------------------------实现一个简单的RNN模型--------------------------------
class My_RNN(Model):
    def __init__(
        self, 
        vocab_size, 
        vec_dim, 
        units, 
        output_dim, 
        activation='tanh',
        **kwargs
        ):
        super(My_RNN, self).__init__(**kwargs)
        self.Embedding = Embedding(vocab_size, vec_dim, input_length=vec_dim)
        #官方实现
        self.SimpleRNN1 = SimpleRNN(units, activation, return_sequences=True)
        self.SimpleRNN2 = SimpleRNN(units, activation, return_sequences=False)
        # 我们自己实现
        #self.SimpleRNN1 = My_SimpleRNN_Layer(units, activation, return_sequences=True)
        #self.SimpleRNN2 = My_SimpleRNN_Layer(units, activation, return_sequences=False)
        self.Dense = Dense(output_dim, activation='sigmoid')

    def call(self, inputs):
        x = self.Embedding(inputs)
        x = self.SimpleRNN1(x)
        x = self.SimpleRNN2(x)
        x = self.Dense(x)
        return x

    def build(self, input_shape):
        super().build(input_shape)
        self.call(Input(input_shape[1:]))
#-----------------------------------------------------------------------------
