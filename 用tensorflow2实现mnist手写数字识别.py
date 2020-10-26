from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense
import numpy as np


# # 2.加载数据集

# 使用tensorflow内置的mnist数据集，返回训练集图片、训练集标签、测试集图片、测试集标签


(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()


# # 3.对图片进行归一化

# 图片每个像素的数值都是在[0, 255]之间，所以归一化要除以255，数据要是浮点数，所以要添加一个小数点


train_images, test_images = train_images / 255.0, test_images / 255.0


# # 4.定义模型

# 搭建一个顺序模型，第一层先将数据展平，原始图片是28x28的灰度图，所以输入尺寸是（28，28），第二层节点数可以自己选择一个合适值，这里用128个节点，激活函数用relu，第三层有多少个种类就写多少，[0, 9]一共有10个数字,所以必须写10，激活函数用softmax


model = keras.Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])


# # 5.指定优化器、损失函数、评价指标

# 优化器使用adam，损失函数使用交叉熵损失函数，评价指标用精确率


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])


# # 6.训练模型

# 将训练集输入模型进行训练，一共训练10次


model.fit(train_images, train_labels, epochs=10)


# # 7.用测试集验证模型效果

# 用测试集去验证训练好的模型，日志等级设置为2


test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Test acc:', test_acc)


# # 8.将图片输入模型，返回预测结果

# 将测试集中的第一张图片输入模型，看是哪个数字的概率最大，并输出真实值

predictions = model.predict(test_images)
print('预测值:', np.argmax(predictions[0]))
print('真实值:', test_labels[0])





