#coding = utf-8
# Import `tensorflow`
import tensorflow as tf
from load import Load
from tfmpELM import ELM
import datetime
import random
import matplotlib.pyplot as plt

# Initialize placeholders
x = tf.placeholder(dtype=tf.float32, shape=[None, 32,32])
y = tf.placeholder(tf.int32, [None])

#path of dataset
path = "F:/gts/gtsdate"
load = Load(path)

# Basic tf setting
tf.set_random_seed(1024)


#
input_len = 1024
output_len = 62
# number of train-image is 4575
batch_size = 4575
hidden_num = 950
wheels_num = 1
#for i in range(1,10,1):
sess = tf.Session()
# Flatten the input data
images_flat = tf.contrib.layers.flatten(load.images32)


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

#######################################2018-5-20 00:10:11######################################################
# testing

test_image_flat = tf.contrib.layers.flatten(load.test_images32)
test_x = sess.run(test_image_flat)
test_y = load.test_labels

sample_indexes = random.sample(range(len(test_x)), 10) 
sample_images = [test_x[i] for i in sample_indexes]
sample_labels = [test_y[i] for i in sample_indexes]

predicted = elm.test(sample_images, sample_labels)
sample_labels_get = sess.run(tf.argmax(sample_labels,1))

print('sample_labels',sample_labels_get)
print('pre',predicted)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_labels_get)):
    truth = sample_labels_get[i]
    prediction = predicted[i]
    plt.subplot(5, 2, 1 + i)
    plt.axis('off')
    color = 'green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
             fontsize=12, color=color)
    plt.imshow(load.test_images32[sample_indexes[i]], cmap="gray")
    plt.imshow(load.test_images32[sample_indexes[i]], cmap="gray")
plt.show()

endtime = datetime.datetime.now()
print('-----------------------------------endtime time---------------------------------------',endtime)
print('time--->',endtime-starttime)
sess.close()
	#wheels_num += 10

	
