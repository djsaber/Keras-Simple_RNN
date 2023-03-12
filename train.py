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
batch_size = 128           # batch����
epochs = 5                 # ѵ������
#-----------------------------------------------------------------------------


#----------------------------------����·��------------------------------------
data_path = "D:/����/python����/�����ֲ�/RNN/datasets/IMDB/imdb.npz"
save_path = "D:/����/python����/�����ֲ�/RNN/save_models/rnn_imdb.h5"
#-----------------------------------------------------------------------------


#----------------------------------�������ݼ�-----------------------------------
(trainX, trainY), (testX, testY) = imdb.load_data(path=data_path, num_words=vocab_size)
print('trainX shape:', trainX.shape)
print('trainY shape:', trainY.shape)
#-----------------------------------------------------------------------------


#---------------------------����Ԥ�����ضϻ���Ϊ�ȳ�---------------------------
trainX = pad_sequences(trainX, maxlen=maxlen)
testX = pad_sequences(testX, maxlen=maxlen)
print('trainX shape:', trainX.shape)
print('trainY shape:', trainY.shape)
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
    optimizer='rmsprop',                 # RMSprop�Ż���
    loss='binary_crossentropy',          # ��Ԫ��������ʧ
    metrics='acc'                        # ��������׼ȷ��
    )
#-----------------------------------------------------------------------------


#-------------------------------ѵ��������ģ��-----------------------------------
rnn.fit(
    trainX, 
    trainY, 
    batch_size=batch_size, 
    epochs=epochs, 
    validation_split=.1            # ȡ10%��������֤
    )
rnn.save_weights(save_path)
#-----------------------------------------------------------------------------