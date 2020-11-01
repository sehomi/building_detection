#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

print("\n\n\n\n\t\t*** START ***\n")

data = np.loadtxt("dataset/association/features/data1.csv", delimiter=",")

X1 = data[:,0:8]
X2 = data[:,8:16]
Y  = data[:,16]

x1_size = X1.shape[1]
x2_size = X2.shape[1]
h1_size = 10
h2_size = 8
y_size  = 1
sgd_step = 0.05
stddev = 0.1
steps = 100

x1 = tf.placeholder("float", shape=[None, x1_size])
x2 = tf.placeholder("float", shape=[None, x2_size])
label  = tf.placeholder("float", shape=[y_size])


weights_1 = tf.Variable(tf.random_normal([x1_size, h1_size], stddev=stddev), name='weights_1')
weights_2 = tf.Variable(tf.random_normal([x2_size, h1_size], stddev=stddev), name='weights_2')
weights_3 = tf.Variable(tf.random_normal([h1_size, h2_size], stddev=stddev), name='weights_3')
weights_4 = tf.Variable(tf.random_normal([h1_size, h2_size], stddev=stddev), name='weights_4')
weights_5 = tf.Variable(tf.random_normal([h2_size, y_size], stddev=stddev), name='weights_5')

b1 = tf.Variable(tf.constant(0.1, shape=(h1_size, 1)), name='b1')
b2 = tf.Variable(tf.constant(0.1, shape=(h1_size, 1)), name='b2')
b3 = tf.Variable(tf.constant(0.1, shape=(h2_size, 1)), name='b3')
b4 = tf.Variable(tf.constant(0.1, shape=(y_size, 1)), name='b4')


#
# h1 = tf.nn.relu( tf.add( tf.matmul(x1, weights_1), tf.transpose(b1) ))
# h2 = tf.nn.relu( tf.add( tf.matmul(x2, weights_2), tf.transpose(b2) ))
# h3 = tf.nn.relu( tf.subtract( tf.add( tf.matmul(h1, weights_3), tf.transpose(b3) ) , tf.add( tf.matmul(h2, weights_4), tf.transpose(b3)) ))

h1 = tf.nn.sigmoid( tf.add( tf.matmul(x1, weights_1), tf.transpose(b1) ))
h2 = tf.nn.sigmoid( tf.add( tf.matmul(x2, weights_2), tf.transpose(b2) ))
h3 = tf.nn.sigmoid( tf.subtract( tf.add( tf.matmul(h1, weights_3), tf.transpose(b3) ) , tf.add( tf.matmul(h2, weights_4), tf.transpose(b3)) ))


y = tf.transpose( tf.nn.softmax( tf.add( tf.matmul(h3, weights_5), tf.transpose(b4) ) ) )

# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=y))
cost = tf.reduce_mean(tf.squared_difference(label, y))
updates_sgd = tf.train.GradientDescentOptimizer(sgd_step).minimize(cost)

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(config=config)
sess.run(init)

for step in range(steps):

    avg_cost = 0.0

    for i in range(len(X1)):
        _, c = sess.run([updates_sgd, cost], feed_dict={x1: X1[i: i + 1], x2: X2[i: i + 1], label: Y[i: i + 1]})
        avg_cost += c

    avg_cost /= X1.shape[0]

    # Print the cost in this epcho to the console.
    if step % 10 == 0:
        print("Epoch: {:3d}    Train Cost: {:.4f}".format(step, avg_cost))
