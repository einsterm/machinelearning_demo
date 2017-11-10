import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

data = pd.read_csv('train.csv')
data = data.fillna(0)
data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
data['Deceased'] = data['Survived'].apply(lambda s: 1 - s)

dataset_X = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']].as_matrix()
dataset_Y = data[['Deceased', 'Survived']].as_matrix()

X_train, X_val, y_train, y_val = train_test_split(dataset_X, dataset_Y,
                                                  test_size=0.2,
                                                  random_state=42)

X = tf.placeholder(tf.float32, shape=[None, 6])
y = tf.placeholder(tf.float32, shape=[None, 2])

weights = tf.Variable(tf.random_normal([6, 2]), name='weights')
bias = tf.Variable(tf.zeros([2]), name='bias')

y_pred = tf.nn.softmax(tf.matmul(X, weights) + bias)

cross_entropy = -tf.reduce_sum(y * tf.log(y_pred + 1e-10), reduction_indices=1)
cost = tf.reduce_mean(cross_entropy)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(10):
        total_loss = 0.
        for i in range(len(X_train)):
            feed_dict = {X: [X_train[i]], y: [y_train[i]]}
            _, loss = sess.run([train_op, cost], feed_dict=feed_dict)
            total_loss += loss
        print('Epoch: %04d, total loss=%.9f' % (epoch + 1, total_loss))
    pred = sess.run(y_pred, feed_dict={X: X_val})
    correct = np.equal(np.argmax(pred, 1), np.argmax(y_val, 1))
    accuracy = np.mean(correct.astype(np.float32))
    print("正确率：%.9f" % accuracy)
