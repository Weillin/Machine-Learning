### 1.导入依赖包


```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense
import numpy as np
import time
```

### 2.加载数据集


```python
(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
```

### 3.图片归一化


```python
train_x, test_x = train_x / 255.0, test_x / 255.0
```

### 4.创建模型


```python
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
```

### 5.指定优化函数、损失函数、评价指标


```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
```

### 6.记录模型开始训练时间


```python
start_time = time.time()
```

### 7.训练模型


```python
model.fit(train_x, train_y, batch_size=64, epochs=10, verbose=1, validation_freq=1)
```

    Epoch 1/10
    938/938 [==============================] - 2s 2ms/step - loss: 0.3128 - accuracy: 0.9122
    Epoch 2/10
    938/938 [==============================] - 2s 2ms/step - loss: 0.1456 - accuracy: 0.9573
    Epoch 3/10
    938/938 [==============================] - 2s 2ms/step - loss: 0.1049 - accuracy: 0.9693
    Epoch 4/10
    938/938 [==============================] - 2s 2ms/step - loss: 0.0809 - accuracy: 0.9767
    Epoch 5/10
    938/938 [==============================] - 2s 2ms/step - loss: 0.0646 - accuracy: 0.9808
    Epoch 6/10
    938/938 [==============================] - 2s 2ms/step - loss: 0.0529 - accuracy: 0.9841
    Epoch 7/10
    938/938 [==============================] - 2s 2ms/step - loss: 0.0445 - accuracy: 0.9871
    Epoch 8/10
    938/938 [==============================] - 2s 2ms/step - loss: 0.0377 - accuracy: 0.9889
    Epoch 9/10
    938/938 [==============================] - 2s 2ms/step - loss: 0.0314 - accuracy: 0.9908
    Epoch 10/10
    938/938 [==============================] - 2s 2ms/step - loss: 0.0263 - accuracy: 0.9926
    




    <tensorflow.python.keras.callbacks.History at 0x21981ca9400>



### 8.记录模型训练完成时间


```python
end_time = time.time()
```

### 9.用测试集验证模型效果


```python
test_loss, test_acc = model.evaluate(test_x, test_y, verbose=1)
predict = model.predict(test_x)
print('time:', end_time - start_time)
print(test_acc)
print('预测值：', np.argmax(predict[0]))
print('真实值：', test_y[0])
```

    313/313 [==============================] - 1s 2ms/step - loss: 0.0698 - accuracy: 0.9797
    time: 19.72472095489502
    0.9797000288963318
    预测值： 7
    真实值： 7
    
