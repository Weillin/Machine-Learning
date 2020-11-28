# ### 导入依赖包
import tensorflow as tf
import matplotlib.pyplot as plt


# ### 加载数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()


# ### 对图片进行归一化
x_train, x_test = x_train / 255.0, x_test / 255.0


# ### 搭建模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),      #Flatten：对图片进行展平    input_shape: 输入图片的尺寸
    tf.keras.layers.Dense(128, activation='relu'),      #Dense: 全连接层    128：神经元个数    activation：激活函数
    tf.keras.layers.Dropout(0.3),                       #Dropout：随机失活神经元，防止过拟合
    tf.keras.layers.Dense(10, activation='softmax')     #10：一共有几个分类
])


# ### 指定优化函数、损失函数、评价指标
model.compile(optimizer='adam',                         #optimizer：优化器
              loss='sparse_categorical_crossentropy',   #loss：损失函数
              metrics=['accuracy'])                     #评价指标

			  
# ### 训练模型
history = model.fit(x_train, y_train,                   #训练数据
                    epochs=10,                          #epochs: 训练次数
                    validation_data=(x_test, y_test),   #validation_data: 验证集
                    verbose=2)                          #verbose：显示日志等级

					
# ### 显示模型准确率的变化过程
plt.plot(history.epoch, history.history.get('accuracy'), label='accuracy')         #横坐标、纵坐标、标签
plt.plot(history.epoch, history.history.get('val_accuracy'), label='val_accuracy')
plt.legend()       #绘图
plt.show()         #显示图像


# ### 显示模型损失的变化过程
plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
plt.legend()
plt.show()
