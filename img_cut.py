# !pip install -q git+https://github.com/tensorflow/examples.git
# !pip install -q -U tfds-nightly

'''导入依赖'''
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow_datasets as tfds
from IPython.display import clear_output
import matplotlib.pyplot as plt

tfds.disable_progress_bar()

'''导入数据集'''
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0  # 归一化
    input_mask -= 1  # 标签减1
    return input_image, input_mask


@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], [128, 128])

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)  # 图像左右翻转
        input_mask = tf.image.flip_left_right(input_mask)  # 轮廓标注左右翻转

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


'''拆分训练集和测试集'''
train_length = info.splits['train'].num_examples
batch_size = 64
buffer_size = 1000
steps_per_epoch = train_length // batch_size

train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(buffer_size).batch(batch_size).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(batch_size)

'''查看原始图片和轮廓标注图片'''


def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


for image, mask in train.take(1):
    sample_image, sample_mask = image, mask
display([sample_image, sample_mask])

'''输出通道为3'''
output_channels = 3

'''使用预训练的MobileNetV2作为编码器/下采样器'''
base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)
layer_names = [
    'block_1_expand_relu'
    , 'block_3_expand_relu'
    , 'block_6_expand_relu'
    , 'block_13_expand_relu'
    , 'block_16_project'
]
layers = [base_model.get_layer(name).output for name in layer_names]
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
down_stack.trainable = False

'''使用pix2pix作为解码器/上采样器'''
up_stack = [
    pix2pix.upsample(512, 3)
    , pix2pix.upsample(256, 3)
    , pix2pix.upsample(128, 3)
    , pix2pix.upsample(64, 3)
]


def unet_model(output_channels):
    inputs = tf.keras.Input(shape=[128, 128, 3])
    x = inputs

    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2
        , padding='same'
    )
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


'''建立模型'''
model = unet_model(output_channels)
model.compile(optimizer='adam'
              , loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
              , metrics=['accuracy'])

'''查看模型结构'''
tf.keras.utils.plot_model(model, show_shapes=True)

'''查看训练前模型的预测结果'''


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
                 create_mask(model.predict(sample_image[tf.newaxis, ...]))])


show_predictions()

'''回调函数'''


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print('Eepoch {}'.format(epoch + 1))


epochs = 20
val_subsplits = 5
validation_steps = info.splits['test'].num_examples // batch_size // val_subsplits

'''训练模型'''
history = model.fit(train_dataset, epochs=epochs
                    , steps_per_epoch=steps_per_epoch
                    , validation_steps=validation_steps
                    , validation_data=test_dataset
                    , callbacks=[DisplayCallback()]
                    )

'''损失可视化'''
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = history.epoch

plt.figure()
plt.plot(epochs, loss, 'g', label='loss')
plt.plot(epochs, val_loss, 'yo', label='val_loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 1])
plt.legend()
plt.show()

'''进行预测'''
show_predictions(test_dataset, 3)
