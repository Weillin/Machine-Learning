import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)

model = tf.keras.Sequential()
model.add(Conv2D(24, kernel_size=5, padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D())

model.add(Conv2D(48, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(48, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(48, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPool2D(padding='same'))

model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPool2D(padding='same'))

model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPool2D(padding='same'))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam'
              , loss='sparse_categorical_crossentropy'
              , metrics=['acc'])

history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

print(history.history.keys())
plt.plot(history.epoch, history.history.get('acc'), label='acc')
plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
plt.legend()
plt.show()
plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
plt.legend()
plt.show()

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Dropout

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)

datagen = ImageDataGenerator(
    rotation_range=10
    , zoom_range=0.10
    , width_shift_range=0.1
    , height_shift_range=0.1)

model = tf.keras.Sequential()

model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size=4, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

datagen.fit(train_images)
history = model.fit(datagen.flow(train_images, train_labels, batch_size=32)
                    , epochs=5, validation_data=(test_images, test_labels))
print(history.history.keys())
plt.plot(history.epoch, history.history.get('acc'), label='acc')
plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
plt.legend()
plt.show()
plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
plt.legend()
plt.show()
