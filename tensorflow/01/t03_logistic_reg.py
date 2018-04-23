# -*- coding: utf-8 -*-
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.examples.tutorials.mnist import input_data
import time

mnist = input_data.read_data_sets('/MNIST_data/', one_hot=True)
# 查看一下数据维度
print(mnist.train.images.shape)
# 查看target维度
print(mnist.train.labels.shape)

batch_size = 128
X = tf.placeholder(tf.float32, [batch_size, 784], name='X')
Y = tf.placeholder(tf.int32, [batch_size, 10], name='Y')

w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name='w')
b = tf.Variable(tf.zeros([1, 10]), name="b")

y_pred = tf.matmul(X, w) + b  # 没有进行softmax激活的值
# 求交叉熵损失
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=Y, name='loss')
# 求平均
loss = tf.reduce_mean(entropy)
learning_rate = 0.01
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 迭代总轮次
n_epochs = 30

with tf.Session() as sess:
    writer = tf.summary.FileWriter('I:/Python35/graphs/logistic_reg', sess.graph)
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    n_batches = int(mnist.train.num_examples / batch_size)
    for i in range(n_epochs):  # 迭代这么多轮
        total_loss = 0
        for _ in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            _, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})
            total_loss += loss_batch
        print('Average loss epoch {0}: {1}'.format(i, total_loss / n_batches))

    print('Total time: {0} seconds'.format(time.time() - start_time))

    print('Optimization Finished!')

    # 测试模型
    y_hat = tf.nn.softmax(y_pred)
    # argmax中的1表示按行求最大值,并返回它的索引
    correct_preds = tf.equal(tf.argmax(y_hat, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))  # 把true和falsh转换成0和1

    n_batches = int(mnist.test.num_examples / batch_size)
    total_correct_preds = 0

    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        accuracy_batch = sess.run([accuracy], feed_dict={X: X_batch, Y: Y_batch})
        total_correct_preds += accuracy_batch[0]

    print('Accuracy {0}'.format(total_correct_preds / mnist.test.num_examples))
    writer.close()
