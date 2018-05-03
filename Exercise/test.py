'''
	test import mnist to project
'''
#import tensorflow.examples.tutorials.mnist.input_data
#mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)