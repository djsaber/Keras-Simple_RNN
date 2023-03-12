# coding=gbk
from keras.datasets import imdb
from keras.utils import pad_sequences
from model import My_RNN


#---------------------------------设置参数-------------------------------------
vocab_size = 20000         # 词库大小
vec_dim = 40               # 词向量维度
maxlen = 80                # 句子最大长度
rnn_units = 32             # rnn特征维度
output_dim = 1             # 输出维度
batch_size = 128           # batch数量
epochs = 5                 # 训练批次
#-----------------------------------------------------------------------------


#----------------------------------设置路径------------------------------------
data_path = "D:/科研/python代码/炼丹手册/RNN/datasets/IMDB/imdb.npz"
save_path = "D:/科研/python代码/炼丹手册/RNN/save_models/rnn_imdb.h5"
#-----------------------------------------------------------------------------


#----------------------------------加载数据集-----------------------------------
(trainX, trainY), (testX, testY) = imdb.load_data(path=data_path, num_words=vocab_size)
print('trainX shape:', trainX.shape)
print('trainY shape:', trainY.shape)
#-----------------------------------------------------------------------------


#---------------------------序列预处理，截断或补齐为等长---------------------------
trainX = pad_sequences(trainX, maxlen=maxlen)
testX = pad_sequences(testX, maxlen=maxlen)
print('trainX shape:', trainX.shape)
print('trainY shape:', trainY.shape)
#-----------------------------------------------------------------------------


#-----------------------------------搭建模型----------------------------------
rnn = My_RNN(
    vocab_size, 
    vec_dim, 
    rnn_units,
    output_dim, 
    )
rnn.build((None, maxlen, ))
rnn.summary()
rnn.compile(
    optimizer='rmsprop',                 # RMSprop优化器
    loss='binary_crossentropy',          # 二元交叉熵损失
    metrics='acc'                        # 计算误差或准确率
    )
#-----------------------------------------------------------------------------


#-------------------------------训练、保存模型-----------------------------------
rnn.fit(
    trainX, 
    trainY, 
    batch_size=batch_size, 
    epochs=epochs, 
    validation_split=.1            # 取10%样本作验证
    )
rnn.save_weights(save_path)
#-----------------------------------------------------------------------------