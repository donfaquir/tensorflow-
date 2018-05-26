#coding=utf-8
from tfmpELM import ELM
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import datetime

# Basic tf setting
tf.set_random_seed(2016)
#
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


#sess=tf.InteractiveSession()

# Get data
mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

# Construct ELM
batch_size = 3000
hidden_num = 200
wheels_num = 1
#for i in range(0,15,1):
sess = tf.Session(config = config)
print("batch_size : {}".format(batch_size))
print("hidden_num : {}".format(hidden_num))
print("wheels_num : {}".format(wheels_num))
starttime = datetime.datetime.now()
print('-------------------------------------------',starttime)
elm = ELM(sess, batch_size, 784, hidden_num, 10,wheels_num)

# one-step feed-forward training
train_x, train_y = mnist.train.next_batch(batch_size)

elm.train(train_x, train_y);

# testing
test_x, test_y = mnist.test.next_batch(batch_size)
elm.test(test_x, test_y)
print('costtime',datetime.datetime.now()-starttime)
sess.close()
#batch_size += 500
#hidden_num +=50