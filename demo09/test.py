import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import os
import time
import glob

img_path = '../images/nobu/'
model_dir = "./model/nobu/"
model_name = "nobunaga_model"

w = 100
h = 100
c = 3


def read_img(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            img = Image.open(im)
            img = img.resize((w, h))
            img = np.array(img)
            imgs.append(img)
            labels.append(idx)
    return np.asanyarray(imgs, np.float32), np.asanyarray(labels, np.int32)


data, label = read_img(img_path)
num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data = data[arr]
label = label[arr]

ratio = 0.8
s = np.int(num_example * ratio)
x_train = data[:s]
y_train = label[:s]
x_val = data[s:]
y_val = label[s:]

x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

conv1 = tf.layers.conv2d(
    inputs=x,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

conv2 = tf.layers.conv2d(

    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu,
    kernel_regularizer=tf.truncated_normal_initializer(stddev=0.01)
)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

conv3 = tf.layers.conv2d(
    inputs=pool2,
    filters=128,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
)
pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

conv4 = tf.layers.conv2d(
    inputs=pool3,
    filters=128,
    kernel_size=[3, 4],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
)

pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
rel = tf.reshape(pool4, [-1, 6 * 6 * 128])

dense1 = tf.layers.dense(
    inputs=rel,
    units=1024,
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003)
)
dense2 = tf.layers.dense(
    inputs=dense1,
    units=1024,
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003)
)
logits = tf.layers.dense(
    inputs=dense2,
    units=5,
    activation=None,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003)
)
loss = tf.losses.sparse_softmax_cross_entropy(label=y_, logits=logits)
train_op = tf.train.AdadeltaOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx, start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield input(excerpt, targets[excerpt])
