import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler

train = pd.read_csv('./data/mnist_train.csv')
test = pd.read_csv('./data/mnist_test.csv')
print(train.shape)
print(test.shape)

train.head(3)

test.head(3)

Y_train = train['label']
X_train = train.drop(labels=['label'], axis=1)
X_train /= 255.0
X_test = test / 255.0
X_train = X_train.values.reshape(-1, 28, 28, 1)
X_test = X_test.values.reshape(-1, 28, 28, 1)
Y_train = to_categorical(Y_train, num_classes=10)

plt.figure(figsize=(15, 5))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(X_train[i].reshape((28, 28)), cmap=plt.cm.binary)
    plt.axis('off')
plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
plt.show()

datagen = ImageDataGenerator(
    rotation_range=10
    , zoom_range=0.01
    , width_shift_range=0.1
    , height_shift_range=0.1
)

X_train1 = X_train[9,].reshape((1, 28, 28, 1))
Y_train1 = Y_train[9,].reshape((1, 10))
plt.figure(figsize=(15, 5))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    X_train2, Y_train2 = datagen.flow(X_train1, Y_train1).next()
    plt.imshow(X_train2[0].reshape((28, 28)), cmap=plt.cm.binary)
    plt.axis('off')
    if i == 9:
        X_train1 = X_train[11,].reshape((1, 28, 28, 1))
    if i == 19:
        X_train1 = X_train[18,].reshape((1, 28, 28, 1))
plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
plt.show()

nets = 15
model = [0] * nets
for i in range(nets):
    model[i] = Sequential()
    model[i].add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    model[i].add(BatchNormalization())
    model[i].add(Conv2D(32, kernel_size=3, activation='relu'))
    model[i].add(BatchNormalization())
    model[i].add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
    model[i].add(BatchNormalization())
    model[i].add(Dropout(0.4))

    model[i].add(Conv2D(64, kernel_size=3, activation='relu'))
    model[i].add(BatchNormalization())
    model[i].add(Conv2D(64, kernel_size=3, activation='relu'))
    model[i].add(BatchNormalization())
    model[i].add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
    model[i].add(BatchNormalization())
    model[i].add(Dropout(0.4))

    model[i].add(Conv2D(128, kernel_size=4, activation='relu'))
    model[i].add(BatchNormalization())
    model[i].add(Flatten())
    model[i].add(Dropout(0.3))
    model[i].add(Dense(10, activation='softmax'))

    model[i].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
history = [0] * nets
epochs = 30
for i in range(nets):
    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size=0.1)
    history[i] = model[i].fit_generator(datagen.flow(X_train2, Y_train2, batch_size=64)
                                        , epochs=epochs
                                        , steps_per_epoch=X_train2.shape[0] // 64
                                        , validation_data=(X_val2, Y_val2)
                                        , callbacks=[annealer]
                                        , verbose=0)
    print('CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}'.format(
        i + 1, epochs, max(history[i].history['accuracy']), max(history[i].history['val_accuracy'])))

results = np.zeros((X_test.shape[0], 10))
for i in range(nets):
    results = results + model[i].predict(X_test)
results = np.argmax(results, axis=1)
results = pd.Series(results, name='Label')
submission = pd.concat([pd.Series(range(1, 28001), name='ImageId'), results], axis=1)
submission.to_csv('./MNIST_ENSEMBLE.csv', index=False)

plt.figure(figsize=(15, 6))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(X_test[i].reshape((28, 28)), cmap=plt.cm.binary)
    plt.title('predict: {}'.format(results[i]), y=0.9)
    plt.axis('off')
plt.subplots_adjust(wspace=0.3, hspace=-0.1)
plt.show()

plt.figure(figsize=(15, 3))
for i, h in enumerate(history):
    i += 1
    plt.subplot(len(history) / 5, 5, i)
    plt.plot(h.epoch, h.history.get('accuracy'), label='acc')
    plt.plot(h.epoch, h.history.get('val_accuracy'), label='val_acc')
    plt.legend()
plt.subplots_adjust(wspace=0.3, hspace=0)
plt.show()

plt.figure(figsize=(15, 3))
for i, h in enumerate(history):
    i += 1
    plt.subplot(len(history) / 5, 5, i)
    plt.plot(h.epoch, h.history.get('loss'), label='loss')
    plt.plot(h.epoch, h.history.get('val_loss'), label='val_loss')
    plt.legend()
plt.subplots_adjust(wspace=0.3, hspace=0)
plt.show()
