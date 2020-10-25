```python
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
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
    26427392/26421880 [==============================] - 21s 1us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
    8192/5148 [===============================================] - 1s 72us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
    4423680/4422102 [==============================] - 5s 1us/step
    Train on 60000 samples
    Epoch 1/10
    60000/60000 [==============================] - 5s 84us/sample - loss: 0.4983 - accuracy: 0.8239
    Epoch 2/10
    60000/60000 [==============================] - 3s 46us/sample - loss: 0.3780 - accuracy: 0.8615
    Epoch 3/10
    60000/60000 [==============================] - 3s 46us/sample - loss: 0.3355 - accuracy: 0.8770
    Epoch 4/10
    60000/60000 [==============================] - 3s 46us/sample - loss: 0.3112 - accuracy: 0.8857
    Epoch 5/10
    60000/60000 [==============================] - 3s 46us/sample - loss: 0.2925 - accuracy: 0.8917
    Epoch 6/10
    60000/60000 [==============================] - 3s 46us/sample - loss: 0.2822 - accuracy: 0.8956
    Epoch 7/10
    60000/60000 [==============================] - 3s 46us/sample - loss: 0.2687 - accuracy: 0.9003
    Epoch 8/10
    60000/60000 [==============================] - 3s 46us/sample - loss: 0.2576 - accuracy: 0.9045
    Epoch 9/10
    60000/60000 [==============================] - 3s 46us/sample - loss: 0.2475 - accuracy: 0.9082
    Epoch 10/10
    60000/60000 [==============================] - 3s 46us/sample - loss: 0.2378 - accuracy: 0.9107
    10000/10000 - 0s - loss: 0.3435 - accuracy: 0.8823
    Test accuracy: 0.8823
    [9.7140310e-07 2.6951807e-10 3.3639935e-08 9.7245267e-10 1.7612262e-10
     7.9476293e-03 8.9016204e-08 3.7341986e-02 1.4517154e-07 9.5470923e-01]
    预测值: 9
    真实值: 9



```python

```


```python

```


```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense
import numpy as np

# 加载数据集
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
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
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11493376/11490434 [==============================] - 8s 1us/step
    Train on 60000 samples
    Epoch 1/10
    60000/60000 [==============================] - 3s 49us/sample - loss: 0.2634 - accuracy: 0.9246
    Epoch 2/10
    60000/60000 [==============================] - 3s 46us/sample - loss: 0.1174 - accuracy: 0.9650
    Epoch 3/10
    60000/60000 [==============================] - 3s 46us/sample - loss: 0.0799 - accuracy: 0.9755
    Epoch 4/10
    60000/60000 [==============================] - 3s 46us/sample - loss: 0.0595 - accuracy: 0.9814
    Epoch 5/10
    60000/60000 [==============================] - 3s 46us/sample - loss: 0.0459 - accuracy: 0.9855
    Epoch 6/10
    60000/60000 [==============================] - 3s 46us/sample - loss: 0.0359 - accuracy: 0.9889
    Epoch 7/10
    60000/60000 [==============================] - 3s 46us/sample - loss: 0.0283 - accuracy: 0.9910
    Epoch 8/10
    60000/60000 [==============================] - 3s 46us/sample - loss: 0.0240 - accuracy: 0.9927
    Epoch 9/10
    60000/60000 [==============================] - 3s 46us/sample - loss: 0.0195 - accuracy: 0.9937
    Epoch 10/10
    60000/60000 [==============================] - 3s 47us/sample - loss: 0.0160 - accuracy: 0.9952
    10000/10000 - 0s - loss: 0.0890 - accuracy: 0.9764
    Test accuracy: 0.9764
    [2.9402461e-10 8.5764768e-10 4.6331972e-10 2.7767308e-07 5.2145441e-13
     4.5972287e-10 2.4680848e-14 9.9999952e-01 1.3935767e-09 2.8545594e-07]
    预测值: 7
    真实值: 7



```python
import matplotlib.pyplot as plt
plt.plot(history.epoch, history.history.get('acc'), label='acc')         #横坐标、纵坐标、标签
plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
plt.legend()       #绘图
plt.show()         #显示图像
# 显示模型损失的变换过程
plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
plt.legend()
plt.show()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-8-400ab90e5e01> in <module>()
          1 import matplotlib.pyplot as plt
    ----> 2 plt.plot(history.epoch, history.history.get('acc'), label='acc')         #横坐标、纵坐标、标签
          3 plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
          4 plt.legend()       #绘图
          5 plt.show()         #显示图像


    NameError: name 'history' is not defined




# 欸哦改名为欧美给我们
而给个摩尔看门狗v
而改为破i结果【文凭
而该机为破格呢
而更为关键二篇【吴干

# 二位股票【可我怕【分数

# 欸哦改名为欧美给我们
而给个摩尔看门狗v
而改为破i结果【文凭
而该机为破格呢
而更为关键二篇【吴干

# 二位股票【可我怕【分数

# 欸哦改名为欧美给我们
而给个摩尔看门狗v
而改为破i结果【文凭
而该机为破格呢
而更为关键二篇【吴干

# 二位股票【可我怕【分数

# 欸哦改名为欧美给我们
而给个摩尔看门狗v
而改为破i结果【文凭
而该机为破格呢
而更为关键二篇【吴干

# 二位股票【可我怕【分数

# 欸哦改名为欧美给我们
而给个摩尔看门狗v
而改为破i结果【文凭
而该机为破格呢
而更为关键二篇【吴干

# 二位股票【可我怕【分数

# 欸哦改名为欧美给我们
而给个摩尔看门狗v
而改为破i结果【文凭
而该机为破格呢
而更为关键二篇【吴干

# 二位股票【可我怕【分数

# 欸哦改名为欧美给我们
而给个摩尔看门狗v
而改为破i结果【文凭
而该机为破格呢
而更为关键二篇【吴干

# 二位股票【可我怕【分数




```python

```
