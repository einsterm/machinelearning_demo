# -*- coding: utf-8 -*-
import tensorflow as tf
from numpy.random import RandomState

# 1.定义神经网络的参数，输入和输出节点。
batch_size = 8
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 2.定义前向传播过程，损失函数及反向传播算法。
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
y = tf.sigmoid(y)

# tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。
# 小于min的让它等于min，大于max的元素的值等于max ，clip_by_value这样做为了不使产生log0
clipVal1 = tf.clip_by_value(y, 1e-10, 1.0)
clipVal2 = tf.clip_by_value(1 - y, 1e-10, 1.0)

# reduce_mean 求平均值
# 交叉熵公式
# tf.log 计算TensorFlow的自然对数
cross_entropy = -tf.reduce_mean(y_ * tf.log(clipVal1) + (1 - y_) * tf.log(clipVal2))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 3.生成模拟数据集。
rdm = RandomState(1)
X = rdm.rand(128, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

# 4.创建一个会话来运行TensorFlow程序。
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 输出目前（未经训练）的参数取值。
    print(sess.run(w1))
    print(sess.run(w2))
    print("\n")

    # 训练模型。
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % 128
        end = (i * batch_size) % 128 + batch_size
        sess.run([train_step, y, y_], feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))

    # 输出训练后的参数取值。
    print("\n")
    print(sess.run(w1))
    print(sess.run(w2))
