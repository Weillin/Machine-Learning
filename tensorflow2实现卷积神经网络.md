### 导入依赖包


```python
import tensorflow as tf
import matplotlib.pyplot as plt
```

### 加载数据集


```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
```

### 对图片进行归一化


```python
x_train, x_test = x_train / 255.0, x_test / 255.0
```

### 搭建模型


```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),      #Flatten：对图片进行展平    input_shape: 输入图片的尺寸
    tf.keras.layers.Dense(128, activation='relu'),      #Dense: 全连接层    128：神经元个数    activation：激活函数
    tf.keras.layers.Dropout(0.3),                       #Dropout：随机失活神经元，防止过拟合
    tf.keras.layers.Dense(10, activation='softmax')     #10：一共有几个分类
])
```

### 指定优化函数、损失函数、评价指标


```python
model.compile(optimizer='adam',                         #optimizer：优化器
              loss='sparse_categorical_crossentropy',   #loss：损失函数
              metrics=['accuracy'])                          #评价指标
```

### 训练模型


```python
history = model.fit(x_train, y_train,                   #训练数据
                    epochs=10,                          #epochs: 训练次数
                    validation_data=(x_test, y_test),   #validation_data: 验证集
                    verbose=2)                          #verbose：显示日志等级
```

    Epoch 1/10
    1875/1875 - 4s - loss: 0.5542 - accuracy: 0.8038 - val_loss: 0.4297 - val_accuracy: 0.8462
    Epoch 2/10
    1875/1875 - 4s - loss: 0.4191 - accuracy: 0.8484 - val_loss: 0.4091 - val_accuracy: 0.8519
    Epoch 3/10
    1875/1875 - 4s - loss: 0.3877 - accuracy: 0.8587 - val_loss: 0.3817 - val_accuracy: 0.8657
    Epoch 4/10
    1875/1875 - 4s - loss: 0.3673 - accuracy: 0.8663 - val_loss: 0.3745 - val_accuracy: 0.8642
    Epoch 5/10
    1875/1875 - 4s - loss: 0.3519 - accuracy: 0.8712 - val_loss: 0.3675 - val_accuracy: 0.8672
    Epoch 6/10
    1875/1875 - 3s - loss: 0.3417 - accuracy: 0.8749 - val_loss: 0.3700 - val_accuracy: 0.8674
    Epoch 7/10
    1875/1875 - 4s - loss: 0.3313 - accuracy: 0.8773 - val_loss: 0.3693 - val_accuracy: 0.8716
    Epoch 8/10
    1875/1875 - 4s - loss: 0.3259 - accuracy: 0.8785 - val_loss: 0.3525 - val_accuracy: 0.8751
    Epoch 9/10
    1875/1875 - 4s - loss: 0.3143 - accuracy: 0.8842 - val_loss: 0.3530 - val_accuracy: 0.8776
    Epoch 10/10
    1875/1875 - 4s - loss: 0.3101 - accuracy: 0.8859 - val_loss: 0.3581 - val_accuracy: 0.8751
    

### 显示模型准确率的变化过程


```python
plt.plot(history.epoch, history.history.get('accuracy'), label='accuracy')         #横坐标、纵坐标、标签
plt.plot(history.epoch, history.history.get('val_accuracy'), label='val_accuracy')
plt.legend()       #绘图
plt.show()         #显示图像
```


![png](output_13_0.png)


### 显示模型损失的变化过程


```python
plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
plt.legend()
plt.show()
```


![png](output_15_0.png)

