from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Dense, Flatten, Dropout, MaxPool2D

'''导入数据'''
train = pd.read_csv('./data/fashion_train.csv')
test = pd.read_csv('./data/fashion_test.csv')
print(train.shape, test.shape)

'''数据预处理'''
input_shape = (28, 28, 1)
x = np.array(train.iloc[:, 1:])
y = keras.utils.to_categorical(np.array(train.iloc[:, 0]))
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
print(x_train.shape, y_train.shape)

x_test = np.array(test.iloc[:, 0:])
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
print(x_train.shape, y_train.shape)

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_val /= 255
x_test /= 255

batch_size = 64
classes = 10
epochs = 5

'''建立模型'''
model = keras.models.Sequential([
    Conv2D(filters=64, kernel_size=(3, 3), padding='same')
    , BatchNormalization()
    , Activation('relu')
    , Conv2D(filters=64, kernel_size=(3, 3), padding='same')
    , BatchNormalization()
    , Activation('relu')
    , MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
    , Dropout(0.2)

    , Conv2D(filters=128, kernel_size=(3, 3), padding='same')
    , BatchNormalization()
    , Activation('relu')
    , Conv2D(filters=128, kernel_size=(3, 3), padding='same')
    , BatchNormalization()
    , Activation('relu')
    , MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
    , Dropout(0.2)

    , Conv2D(filters=256, kernel_size=(3, 3), padding='same')
    , BatchNormalization()
    , Activation('relu')
    , Conv2D(filters=256, kernel_size=(3, 3), padding='same')
    , BatchNormalization()
    , Activation('relu')
    , Conv2D(filters=256, kernel_size=(3, 3), padding='same')
    , BatchNormalization()
    , Activation('relu')
    , MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
    , Dropout(0.2)

    , Conv2D(filters=512, kernel_size=(3, 3), padding='same')
    , BatchNormalization()
    , Activation('relu')
    , Conv2D(filters=512, kernel_size=(3, 3), padding='same')
    , BatchNormalization()
    , Activation('relu')
    , Conv2D(filters=512, kernel_size=(3, 3), padding='same')
    , BatchNormalization()
    , Activation('relu')
    , MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
    , Dropout(0.2)

    , Conv2D(filters=512, kernel_size=(3, 3), padding='same')
    , BatchNormalization()
    , Activation('relu')
    , Conv2D(filters=512, kernel_size=(3, 3), padding='same')
    , BatchNormalization()
    , Activation('relu')
    , Conv2D(filters=512, kernel_size=(3, 3), padding='same')
    , BatchNormalization()
    , Activation('relu')
    , MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
    , Dropout(0.2)

    , Flatten()
    , Dense(512, activation='relu')
    , Dropout(0.2)
    , Dense(512, activation='relu')
    , Dropout(0.2)
    , Dense(classes, activation='softmax')
])

model.compile(optimizer='adam'
              , loss='categorical_crossentropy'
              , metrics=['accuracy'])

'''断点续训'''
save_path = './checkpoint/VGG16.ckpt'
if os.path.exists(save_path + '.index'):
    print('model loading')
    model.load_weights(save_path)
cp_callback = keras.callbacks.ModelCheckpoint(filepath=save_path
                                              , save_weights_only=True
                                              , save_best_only=True)

'''训练模型'''
history = model.fit(x_train, y_train
                    , batch_size=batch_size
                    , epochs=epochs
                    , verbose=1
                    , validation_data=(x_val, y_val)
                    , callbacks=[cp_callback])

'''预测结果'''
result = model.predict(x_test)
pred = tf.argmax(result, axis=1)
df = pd.DataFrame(pred, columns=['label'])
df.to_csv(path_or_buf='Submission.csv', index_label='image_id')

'''损失和准确率可视化'''
print(history.history.keys())
plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
plt.legend()
plt.show()

plt.plot(history.epoch, history.history.get('accuracy'), label='acc')
plt.plot(history.epoch, history.history.get('val_accuracy'), label='val_acc')
plt.legend()
plt.show()
