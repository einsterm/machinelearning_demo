# -*- coding: utf-8 -*-
import tensorflow as tf

step = 100
global_step = tf.Variable(0)  # 最终不能少于0
start_rate = 0.1
# 每迭代5次，rate乘以0.96
rate = tf.train.exponential_decay(start_rate, global_step, 5, 0.96, staircase=True)

x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x")
y = tf.square(x)
train_op = tf.train.GradientDescentOptimizer(rate).minimize(y, global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(step):
        sess.run(train_op)
        if i % 5 == 0:
            LEARNING_RATE_value = sess.run(rate)
            x_value = sess.run(x)
            print("After %s iteration(s): x%s is %f, learning rate is %f." % (i + 1, i + 1, x_value, LEARNING_RATE_value))
