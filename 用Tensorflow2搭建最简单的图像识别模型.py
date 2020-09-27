# 安装tensorflow2.2.0版本，我这个是CPU版，也可以安装GPU版本。GPU版本必须是NVIDIA显卡，而且要需要安装CUDA和cuDNN
# pip install tensorflow-cpu==2.2.0 -i https://pypi.douban.com/simple/          #使用豆瓣镜像源安装会快很多

# 导入依赖包
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense
import numpy as np

# 加载数据集
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# 对图片进行归一化
train_images = train_images / 255.0
test_images = test_images / 255.0
# 定义模型
model = keras.Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
# 指定优化函数、损失函数、评价指标
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 训练模型
model.fit(train_images, train_labels, epochs=10)
# 用测试集验证模型效果
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Test accuracy:', test_acc)
# 将图片输入模型，返回预测结果
predictions = model.predict(test_images)
print(predictions[0])
print('预测值:', np.argmax(predictions[0]))
print('真实值:', test_labels[0])
