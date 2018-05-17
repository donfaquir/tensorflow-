#coding = utf-8
# Import `tensorflow`
import tensorflow as tf
from load import Load
from tfmpELM import ELM
import datetime

# Initialize placeholders
x = tf.placeholder(dtype=tf.float32, shape=[None, 32,32])
y = tf.placeholder(tf.int32, [None])

#path of dataset
path = "F:/gts/gtsdate"
load = Load(path)

# Basic tf setting
tf.set_random_seed(1024)
sess = tf.Session()

#
input_len = 1024
output_len = 62
# number of train-image is 4575
batch_size = 4575
hidden_num = 1200
wheels_num = 1

# Flatten the input data
images_flat = tf.contrib.layers.flatten(load.images32)
test_image_flat = tf.contrib.layers.flatten(load.test_images32)

print()
print('-----------------------------------start time---------------------------------------',datetime.datetime.now())
print("batch_size : {}".format(batch_size))
print("hidden_num : {}".format(hidden_num))
print("wheels_num : {}".format(wheels_num))
starttime = datetime.datetime.now()
elm = ELM(sess, batch_size, input_len, hidden_num, output_len,wheels_num)

# one-step feed-forward training
train_x = sess.run(images_flat)
train_y = load.labels
elm.train(train_x, train_y)

# testing
test_x = sess.run(test_image_flat)
test_y = load.test_labels
elm.test(test_x, test_y)

endtime = datetime.datetime.now()
print('time--->',endtime-starttime)
