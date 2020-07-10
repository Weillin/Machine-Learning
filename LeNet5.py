from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras import models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.set_printoptions(threshold=np.inf)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''加载和处理本地数据'''


def load_data():
    x_list, y_list = [], []
    class_map = {'man': '0', 'dog': '1', 'cat': '2'}

    data_path = './data/'
    x_train_path = './x_train.npy'
    y_train_path = './y_train.npy'
    x_test_path = './x_test.npy'
    y_test_path = './y_test.npy'

    # 如果本地有npy数据直接加载,避免多次读取本地图片
    if os.path.exists(x_train_path) and os.path.exists(y_train_path) and \
            os.path.exists(x_test_path) and os.path.exists(y_test_path):
        print('Data loading')
        x_train = np.load(x_train_path)
        y_train = np.load(y_train_path)
        x_test = np.load(x_test_path)
        y_test = np.load(y_test_path)
    # 处理本地图片
    else:
        L = []
        for dirpath, dirnames, filenames in os.walk(data_path):
            for file in filenames:
                if os.path.splitext(file)[-1] == '.jpg':
                    L.append(os.path.join(dirpath, file))
        i = 0
        for filename in L:
            value = re.split('\\\|\.', filename)
            img_path = filename
            img = Image.open(img_path).convert('RGB')  # 以'RGB'模式读取图片
            img = img.resize((32, 32), Image.ANTIALIAS)
            img = np.array(img)
            # img = img / 255.   #图片归一化,可以在加载时处理,也可以在数据增强时处理
            x_list.append(img)
            y = value[-3]
            y = int(class_map.get(y))
            y_list.append(y)
            print(i, img_path)
            i += 1

        x = np.array(x_list)
        y = np.array(y_list)

        y = y.astype(np.int64)
        y = y.reshape(-1, 1)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

        np.save(x_train_path, x_train)
        np.save(y_train_path, y_train)
        np.save(x_test_path, x_test)
        np.save(y_test_path, y_test)

    return x_train, y_train, x_test, y_test


'''建立模型'''


def LeNet5(x_train, y_train, x_test, y_test):
    model = models.Sequential([
        Conv2D(filters=6, kernel_size=(5, 5), activation='sigmoid', input_shape=(32, 32, 3))
        , MaxPool2D(pool_size=(2, 2), strides=2)
        , Conv2D(filters=16, kernel_size=(5, 5), activation='sigmoid')
        , MaxPool2D(pool_size=(2, 2), strides=2)
        , Flatten()
        , Dense(120, activation='sigmoid')
        , Dense(84, activation='sigmoid')
        , Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )
    # 断点续训
    save_path = './checkpoint/LeNet5.ckpt'
    if os.path.exists(save_path + '.index'):
        print('model loading')
        model.load_weights(save_path)
    cp_callback = callbacks.ModelCheckpoint(filepath=save_path
                                            , save_weights_only=True
                                            , save_best_only=True)
    log_dir = os.path.join('logs')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # 数据增强,本例中不使用效果会更好
    image_gen_train = ImageDataGenerator(
        rescale=1. / 255.
        , rotation_range=45
        , width_shift_range=.15
        , height_shift_range=.15
        , horizontal_flip=True
        , zoom_range=0.5
    )
    image_gen_train.fit(x_train)

    history = model.fit(image_gen_train.flow(x_train, y_train, batch_size=32), epochs=5
                        , validation_data=(x_test, y_test)
                        , callbacks=[cp_callback, tensorboard_callback])

    # 参数提取
    with open('./weights.txt', 'w') as f:
        for v in model.trainable_variables:
            f.write(str(v.name) + '\n')
            f.write(str(v.shape) + '\n')
            f.write(str(v.numpy()) + '\n')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    return acc, val_acc, loss, val_loss


'''可视化准确率和损失'''


def result_show(acc, val_acc, loss, val_loss):
    # 查看准确率
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train acc')
    plt.plot(val_acc, label='Vali acc')
    plt.title('Train and Vali acc')
    plt.legend()

    # 查看损失
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train loss')
    plt.plot(val_loss, label='Vali loss')
    plt.title('Train and Vali loss')
    plt.legend()
    plt.show()
    # 浏览器查看命令
    # tensorboard --logdir="logs"


def main():
    x_train, y_train, x_test, y_test = load_data()
    acc, val_acc, loss, val_loss = LeNet5(x_train, y_train, x_test, y_test)
    result_show(acc, val_acc, loss, val_loss)


if __name__ == '__main__':
    main()
