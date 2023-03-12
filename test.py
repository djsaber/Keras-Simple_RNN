# coding=gbk

from keras.datasets import imdb
from keras.utils import pad_sequences
from model import My_RNN



#---------------------------------���ò���-------------------------------------
vocab_size = 20000         # �ʿ��С
vec_dim = 40               # ������ά��
maxlen = 80                # ������󳤶�
rnn_units = 32             # rnn����ά��
output_dim = 1             # ���ά��
batch_size = 32
#-----------------------------------------------------------------------------


#----------------------------------����·��------------------------------------
data_path = "D:/����/python����/�����ֲ�/RNN/datasets/IMDB/imdb.npz"
load_path = "D:/����/python����/�����ֲ�/RNN/save_models/rnn_imdb.h5"
#-----------------------------------------------------------------------------


#----------------------------------�������ݼ�-----------------------------------
_, (testX, testY) = imdb.load_data(path=data_path, num_words=vocab_size)
print('testX shape:', testX.shape)
print('testY shape:', testY.shape)
#-----------------------------------------------------------------------------


#---------------------------����Ԥ�����ضϻ���Ϊ�ȳ�---------------------------
testX = pad_sequences(testX, maxlen=maxlen)
print('trainX shape:', testX.shape)
#-----------------------------------------------------------------------------


#-----------------------------------�ģ��----------------------------------
rnn = My_RNN(
    vocab_size, 
    vec_dim, 
    rnn_units,
    output_dim, 
    )
rnn.build((None, maxlen, ))
rnn.summary()
rnn.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics='acc'
    )
rnn.load_weights(load_path)
#-----------------------------------------------------------------------------


#-------------------------------ѵ��������ģ��-----------------------------------
rnn.evaluate(
    testX, 
    testY, 
    batch_size=batch_size, 
    )
#-----------------------------------------------------------------------------