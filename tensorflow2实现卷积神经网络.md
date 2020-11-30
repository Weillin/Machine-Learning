### 1.导入依赖包


```python
import tensorflow as tf
import matplotlib.pyplot as plt
```

### 2.加载数据集


```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
```

### 3.对图片进行归一化


```python
x_train, x_test = x_train / 255.0, x_test / 255.0
```

### 4.搭建模型


```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),      #Flatten：对图片进行展平    input_shape: 输入图片的尺寸
    tf.keras.layers.Dense(128, activation='relu'),      #Dense: 全连接层    128：神经元个数    activation：激活函数
    tf.keras.layers.Dropout(0.4),                       #Dropout：随机失活神经元，防止过拟合
    tf.keras.layers.Dense(10, activation='softmax')     #10：一共有几个分类
])
```

### 5.指定优化函数、损失函数、评价指标


```python
model.compile(optimizer='adam',                         #optimizer：优化器
              loss='sparse_categorical_crossentropy',   #loss：损失函数
              metrics=['accuracy'])                          #评价指标
```

### 5.训练模型


```python
history = model.fit(x_train, y_train,                   #训练数据
                    epochs=10,                          #epochs: 训练次数
                    validation_data=(x_test, y_test),   #validation_data: 验证集
                    verbose=2)                          #verbose：显示日志等级
```

    Epoch 1/10
    1875/1875 - 4s - loss: 0.5850 - accuracy: 0.7926 - val_loss: 0.4463 - val_accuracy: 0.8377
    Epoch 2/10
    1875/1875 - 4s - loss: 0.4438 - accuracy: 0.8404 - val_loss: 0.4177 - val_accuracy: 0.8502
    Epoch 3/10
    1875/1875 - 4s - loss: 0.4103 - accuracy: 0.8515 - val_loss: 0.3987 - val_accuracy: 0.8559
    Epoch 4/10
    1875/1875 - 4s - loss: 0.3888 - accuracy: 0.8584 - val_loss: 0.3770 - val_accuracy: 0.8623
    Epoch 5/10
    1875/1875 - 4s - loss: 0.3746 - accuracy: 0.8644 - val_loss: 0.3693 - val_accuracy: 0.8662
    Epoch 6/10
    1875/1875 - 4s - loss: 0.3635 - accuracy: 0.8662 - val_loss: 0.3670 - val_accuracy: 0.8680
    Epoch 7/10
    1875/1875 - 4s - loss: 0.3524 - accuracy: 0.8693 - val_loss: 0.3592 - val_accuracy: 0.8689
    Epoch 8/10
    1875/1875 - 4s - loss: 0.3457 - accuracy: 0.8730 - val_loss: 0.3568 - val_accuracy: 0.8724
    Epoch 9/10
    1875/1875 - 4s - loss: 0.3386 - accuracy: 0.8748 - val_loss: 0.3412 - val_accuracy: 0.8775
    Epoch 10/10
    1875/1875 - 4s - loss: 0.3349 - accuracy: 0.8753 - val_loss: 0.3503 - val_accuracy: 0.8738
    

### 6.显示模型准确率的变化过程


```python
plt.plot(history.epoch, history.history.get('accuracy'), label='accuracy')         #横坐标、纵坐标、标签
plt.plot(history.epoch, history.history.get('val_accuracy'), label='val_accuracy')
plt.legend()       #绘图
plt.show()         #显示图像
```


![png](output_13_0.png)


### 7.显示模型损失的变化过程


```python
plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
plt.legend()
plt.show()
```


![png](output_15_0.png)

