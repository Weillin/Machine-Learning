# 导入依赖包
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense
import numpy as np
import time

# 加载数据集
(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
# 图片归一化
train_x, test_x = train_x / 255.0, test_x / 255.0

# 创建模型
class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = Flatten(input_shape=(28, 28))
        self.dense2 = Dense(100, activation='relu')
        self.dense3 = Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


model = MyModel()
# 指定优化函数、损失函数、评价指标
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
# 记录模型开始训练时间
start_time = time.time()
# 训练模型
model.fit(train_x, train_y, batch_size=64, epochs=10, verbose=1, validation_freq=1)
# 记录模型训练完成时间
end_time = time.time()

# 用测试集验证模型效果
test_loss, test_acc = model.evaluate(test_x, test_y, verbose=1)
predict = model.predict(test_x)
print('time:', end_time - start_time)
print(test_acc)
print('预测值：', np.argmax(predict[0]))
print('真实值：', test_y[0])
