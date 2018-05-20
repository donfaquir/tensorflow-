#coding=utf-8
from tfELM import ELM
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import datetime

# Basic tf setting
tf.set_random_seed(2016)
sess = tf.Session()

# Get data
mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

# Construct ELM
batch_size = 5000
hidden_num = 40
wheels_num = 1
input_len = 784
output_len = 10
print()
print('-----------------------------------start time---------------------------------------',datetime.datetime.now())
print("batch_size : {}".format(batch_size))
print("hidden_num : {}".format(hidden_num))
print("wheels_num : {}".format(wheels_num))
elm = ELM(sess, batch_size, input_len, hidden_num, output_len,wheels_num)

# one-step feed-forward training
train_x, train_y = mnist.train.next_batch(batch_size)

elm.train(train_x, train_y);

# testing
test_x, test_y = mnist.test.next_batch(batch_size)
elm.test(test_x, test_y)
print('-------------------------------end time-------------------------------------------',datetime.datetime.now())