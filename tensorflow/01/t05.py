# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/MNIST_data/', one_hot=True)
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels

n_input = 784
n_output = 10
# [w,h,c,n] c表示颜色通道个数，这里只有灰度，所以为1，n表示filter个数
# 7 * 7 * 128 表示第一个全连接层的输入
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.1)),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.1)),
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 128, 1024], stddev=0.1)),
    'wd2': tf.Variable(tf.random_normal([1024, n_output], stddev=0.1))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64], stddev=0.1)),
    'bc2': tf.Variable(tf.random_normal([128], stddev=0.1)),
    'bd1': tf.Variable(tf.random_normal([1024], stddev=0.1)),
    'bd2': tf.Variable(tf.random_normal([n_output], stddev=0.1))
}


def conv_basic(_input, _w, _b, _keepratio):
    # input[batchSize,w,h,channelSize] -1表示自动计算
    _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1])

    # conv layer 1 这里的padding可以自动计算出来，所以不用指定,SAME是指缺位补0
    _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1, _b['bc1']))
    _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr1 = tf.nn.dropout(_pool1, _keepratio)

    # conv layer 2
    _conv2 = tf.nn.conv2d(_pool_dr1, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b['bc2']))
    _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)

    # vectorize
    _dense1 = tf.reshape(_pool_dr2, [-1, _w['wd1'].get_shape().as_list()[0]])

    # fully connected layer 1
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
    _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)

    # fully connected layer 2
    _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])

    # RETURN
    out = {'input_r': _input_r, 'conv1': _conv1, 'pool1': _pool1, 'pool1_dr1': _pool_dr1,
           'conv2': _conv2, 'pool2': _pool2, 'pool_dr2': _pool_dr2, 'dense1': _dense1,
           'fc1': _fc1, 'fc_dr1': _fc_dr1, 'out': _out
           }
    return out


print("cnn ready")

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keepratio = tf.placeholder(tf.float32)

# FUNCTIONS

_pred = conv_basic(x, weights, biases, keepratio)['out']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(_pred, y))
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
_corr = tf.equal(tf.argmax(_pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(_corr, tf.float32))
init = tf.global_variables_initializer()

# SAVER
print("GRAPH READY")

sess = tf.Session()
sess.run(init)

training_epochs = 15
batch_size = 16
display_step = 1
for epoch in range(training_epochs):
    avg_cost = 0.
    # total_batch = int(mnist.train.num_examples/batch_size)
    total_batch = 10
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Fit training using batch data
        sess.run(optm, feed_dict={x: batch_xs, y: batch_ys, keepratio: 0.7})
        # Compute average loss
        avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keepratio: 1.}) / total_batch

    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys, keepratio: 1.})
        print(" Training accuracy: %.3f" % (train_acc))
        # test_acc = sess.run(accr, feed_dict={x: testimg, y: testlabel, keepratio:1.})
        # print (" Test accuracy: %.3f" % (test_acc))

print("OPTIMIZATION FINISHED")
