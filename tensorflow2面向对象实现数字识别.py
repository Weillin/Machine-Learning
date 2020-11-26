#!/usr/bin/env python
# coding: utf-8

# ### 1.导入依赖包

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense
import numpy as np
import time


# ### 2.加载数据集

# In[2]:


(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()


# ### 3.图片归一化

# In[3]:


train_x, test_x = train_x / 255.0, test_x / 255.0


# ### 4.创建模型

# In[4]:


class Mnist(tf.keras.Model):

    def __init__(self):
        super(Mnist, self).__init__()
        self.dense1 = Flatten(input_shape=(28, 28))
        self.dense2 = Dense(100, activation='relu')
        self.dense3 = Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


model = Mnist()


# ### 5.指定优化函数、损失函数、评价指标

# In[5]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


# ### 6.记录模型开始训练时间

# In[6]:


start_time = time.time()


# ### 7.训练模型

# In[7]:


model.fit(train_x, train_y, batch_size=64, epochs=10, verbose=1, validation_freq=1)


# ### 8.记录模型训练完成时间

# In[8]:


end_time = time.time()


# ### 9.用测试集验证模型效果

# In[9]:


test_loss, test_acc = model.evaluate(test_x, test_y, verbose=1)
predict = model.predict(test_x)
print('time:', end_time - start_time)
print(test_acc)
print('预测值：', np.argmax(predict[0]))
print('真实值：', test_y[0])

