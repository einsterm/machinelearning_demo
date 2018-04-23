from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 1.设置输入和输出节点的个数, 配置神经网络的参数。
input_node = 784  # 输入节点
output_node = 10  # 输出节点
layer1_node = 500  # 隐藏层数

batch_size = 100  # 每次batch打包的样本个数
# 模型相关的参数
learning_rate_base = 0.8
learning_rate_decay = 0.99
regularaztion_rate = 0.0001
training_steps = 5000
moving_average_decay = 0.99


# 2.定义辅助函数来计算前向传播结果，使用ReLU做为激活函数。
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 不使用滑动平均类
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        # 使用滑动平均类
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
    return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


# 3.定义训练过程。
def train(mnist):
    x = tf.placeholder(tf.float32, [None, input_node], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, output_node], name='y-input')

    # 生成隐藏层的参数。
    weights1 = tf.Variable(tf.truncated_normal([input_node, layer1_node], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[layer1_node]))

    # 生成输出层的参数。
    weights2 = tf.Variable(tf.truncated_normal([layer1_node, output_node], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[output_node]))

    # 计算不含滑动平均类的前向传播结果
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义训练轮数及相关的滑动平均类
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵及其平均值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 损失函数的计算
    regularizer = tf.contrib.layers.l2_regularizer(regularaztion_rate)
    regularaztion = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularaztion

    # 设置指数衰减的学习率。
    learning_rate = tf.train.exponential_decay(
        learning_rate_base,
        global_step,
        mnist.train.num_examples / batch_size,
        learning_rate_decay,
        staircase=True)

    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 反向传播更新参数和更新每一个参数的滑动平均值
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
    # 计算正确率
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话，并开始训练过程。
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 循环的训练神经网络。
        for i in range(training_steps):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))

            xs, ys = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
    test_acc = sess.run(accuracy, feed_dict=test_feed)
    print(("After %d training step(s), test accuracy using average model is %g" % (training_steps, test_acc)))

def main(argv=None):
    mnist = input_data.read_data_sets("/MNIST_data/", one_hot=True)
    train(mnist)

# 4.主程序入口，这里设定模型训练次数为5000次。
if __name__ == '__main__':
    main()



