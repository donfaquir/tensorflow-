#coding=utf-8
import tensorflow as tf
import numpy as np
from six import integer_types, string_types
from svd2MP import SVD
import datetime
import os

omega = 1.
class ELM(object):

	def __init__(self, sess, batch_size, input_len, hidden_num, output_len,wheels_num,activation='sigmoid'):
		'''
		Args:
		  sess : TensorFlow session.
		  batch_size : The batch size (N)
		  input_len : The length of input. (L)
		  hidden_num : The number of hidden node. (K)
		  output_len : The length of output. (O)
		'''
		assert isinstance(input_len, integer_types), "Number of inputs must be integer"
		assert isinstance(output_len, integer_types), "Number of outputs must be integer"
		assert batch_size > 0, "batch_size should be positive"
		
		self._sess = sess 
		self._batch_size = batch_size
		self._input_len = input_len
		self._hidden_num = hidden_num
		self._output_len = output_len 
		self._wheels_num = wheels_num
		
		#ensure activation function
		if activation == 'sigmoid':
			self._activation = tf.nn.sigmoid
		elif activation == 'linear' or activation == None:
			self._activation = tf.identity
		elif activation == 'tanh':
			self._activation = tf.tanh
		else:
			raise ValueError(
				'an unknown activation function \'%s\' was given.' % (activation)
			)

		self._x = tf.placeholder(tf.float32, shape=(self._batch_size, self._input_len), name='x')
		self._t = tf.placeholder(tf.float32, shape=(self._batch_size, self._output_len), name='t')
		self._inputW = tf.get_variable(
			'inputW',
			shape=[self._input_len, self._hidden_num],
			initializer=tf.random_uniform_initializer(-1,1),
			trainable=False,
		)
		self._bias = tf.get_variable(
			'bias',
			shape=[self._hidden_num],
			initializer=tf.random_uniform_initializer(-1,1),
			trainable=False,
		)
		self._outputW = tf.get_variable(
			'outputW',
			shape=[self._hidden_num, self._output_len],
			initializer=tf.zeros_initializer(),
			trainable=False,
		)
		
		self._H = tf.matmul(self._x, self._inputW) + self._bias # N x L
		self._svd = SVD(self._sess,self._H)
		self._outputW = tf.matmul(self._svd.get_MP(),self._t) #B = H_T*T
		self._predict = tf.matmul(self._activation(tf.matmul(self._x, self._inputW) + self._bias), self._outputW)
		
		#self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self._predict, labels=self._t))
		self._correct_prediction = tf.equal(tf.argmax(self._predict,1), tf.argmax(self._t,1))
		self._accuracy = tf.reduce_mean(tf.cast(self._correct_prediction, tf.float32))
		
		# Finish initial
		self._is_trained = False
		# Saver
		self._saver = tf.train.Saver()
	
	def train(self, x, t):
		'''
		Args :
			x : input array (N x L)
			t : label array (N x O)
		'''
		if not self._is_trained : 
			
			# Initialize variables
			self._sess.run(tf.global_variables_initializer())
			#self._sess.run(self.H, feed_dict={self._x: x})
			
			#self._sess.run(self._outputW, {self._x:x, self._t:t})
			#self._sess.run(self._predict, feed_dict={self._x: x})
			self._max_accuracy = self._sess.run(self._accuracy,{self._x:x, self._t:t})
			print('first accuracy-->',self._max_accuracy)
			
			#save_path = self._saver.save(self._sess,"./save/model.ckpt")
			#print("Model save in->",save_path)
			i=0
			temp = self._wheels_num/5
			while(i < self._wheels_num):
				i += 1
				self._sess.run(tf.global_variables_initializer())
				#self._sess.run(self._assign_outputW, {self._x:x, self._t:t})
				#self._sess.run(self._predict, feed_dict={self._x: x})
				acc = self._sess.run(self._accuracy,{self._x:x, self._t:t})
				if(acc > self._max_accuracy):
					self._max_accuracy = acc
					#save_path = self._saver.save(self._sess,"./save/model.ckpt")
					#print("Model save in->",save_path)
					print('exchange happened')
				
				if(i%temp == 0):
					print(i,'training-->',datetime.datetime.now())
			print("Train Accuracy: {:.9f}".format(self._max_accuracy))
			self._is_trained  = True
		
	def test(self,x,t):
		if not self._is_trained:
			raise Exception(
				'Please train the neural network first. '
				'please call \'train\' methods for initialization.'
			)
		else:
			print("Test Accuracy: {:.9f}".format(self._sess.run(self._accuracy,{self._x:x, self._t:t})))
		
		
