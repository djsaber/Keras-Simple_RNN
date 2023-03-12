# Keras-Simple_RNN
基于Keras搭建一个简单的RNN，用IMDB影评数据集对RNN进行训练，完成模型的保存和加载和测试。

环境：<br />
CUDA：11.6.134<br />
cuDNN：8.4.0<br />
keras：2.9.0<br />
tensorflow：2.9.1<br /><br />

注意：<br />
项目内目录中两个文件夹：<br />
1. /datasets：存放数据集文件<br />
2. /save_models：保存训练好的模型权重文件<br /><br />

实现自己的自定义RNN：<br />
Keras实现自定循环神经网络需要：
1.实现RnnCell，注意需要定义rnn的状态参数维度：self.state_size<br />
2.将实现好的RnnCell作为参数cell传入RNN()，让Keras自动推断每个状态的传递过程<br /><br />

数据集：<br />
IMDB：影评数据集,训练集/测试集包含25000/25000条影评数据<br />
链接：https://pan.baidu.com/s/18nX-2mqJzYU8XKQ5cfhxvw?pwd=52dl 提取码：52dl<br /><br />

通过对训练集切分10%比例用于训练时验证模型<br />
训练好的模型对测试集进行测试评价效果<br />
