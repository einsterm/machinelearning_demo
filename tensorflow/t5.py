# -*- coding: utf-8 -*-
import tensorflow as tf

step = 10
rate = 1
x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x")
y = tf.square(x)  # 平方 y=x^2

train_op = tf.train.GradientDescentOptimizer(rate).minimize(y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(step):
        sess.run(train_op)
        x_value = sess.run(x)
        print("After %s iteration(s): x%s is %f." % (i + 1, i + 1, x_value))
