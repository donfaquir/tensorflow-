#coding=utf-8
import tensorflow as tf
import numpy as np
from six import integer_types, string_types
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
			self.__activation = tf.nn.sigmoid
		elif activation == 'linear' or activation == None:
			self.__activation = tf.identity
		elif activation == 'tanh':
			self.__activation = tf.tanh
		else:
			raise ValueError(
				'an unknown activation function \'%s\' was given.' % (activation)
			)

		self.__x = tf.placeholder(tf.float32, shape=(self._batch_size, self._input_len), name='x')
		self.__t = tf.placeholder(tf.float32, shape=(self._batch_size, self._output_len), name='t')
		
		self.__inputW = tf.Variable(
			tf.random_normal([self._input_len, self._hidden_num]),
			trainable=False, dtype=tf.float32)
		self.__bias = tf.Variable(
			tf.random_normal([self._hidden_num]),
			trainable=False, dtype=tf.float32)
		self.__outputW = tf.Variable(
			tf.zeros([self._hidden_num, self._output_len]),
			trainable=False, dtype=tf.float32)
		
		'''
		
		self.__inputW = tf.get_variable(
			'inputW',
			shape=[self._input_len, self._hidden_num],
			initializer=tf.random_uniform_initializer(-1,1),
			trainable=False,
		)
		self.__bias = tf.get_variable(
			'bias',
			shape=[self._hidden_num],
			initializer=tf.random_uniform_initializer(-1,1),
			trainable=False,
		)
		self.__outputW = tf.get_variable(
			'outputW',
			shape=[self._hidden_num, self._output_len],
			initializer=tf.zeros_initializer(),
			trainable=False,
		)
		
		'''
		# Predict
		#self.__predict = tf.matmul(self.__activation(tf.matmul(self.__x, self.__inputW) + self.__bias), self.__outputW)
		self.__predict = tf.matmul((tf.matmul(self.__x, self.__inputW) + self.__bias), self.__outputW)
		
		self.H0 = tf.matmul(self.__x, self.__inputW) + self.__bias # N x L
		self.H0_T = tf.transpose(self.H0)
		
		# beta analytic solution : self._beta_s (K x O)
		if self._input_len < self._hidden_num: # L < K
		  identity = tf.constant(np.identity(self._hidden_num), dtype=tf.float32)
		  self._beta_s = tf.matmul(tf.matmul(tf.matrix_inverse(
			tf.matmul(self.H0_T, self.H0) + identity/omega), 
			self.H0_T), self.__t)
		  # _beta_s = (H_T*H + I/om)^(-1)*H_T*T
		else:
		  identity = tf.constant(np.identity(self._batch_size), dtype=tf.float32)
		  self._beta_s = tf.matmul(tf.matmul(self.H0_T, tf.matrix_inverse(
			tf.matmul(self.H0, self.H0_T)+identity/omega)), self.__t)
		  # _beta_s = H_T*(H*H_T + I/om)^(-1)*T
		self._assign_outputW = self.__outputW.assign(self._beta_s)
		
		self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.__predict, labels=self.__t))
		self._correct_prediction = tf.equal(tf.argmax(self.__predict,1), tf.argmax(self.__t,1)) 
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
			self._sess.run(self._assign_outputW, {self.__x:x, self.__t:t})
			self._sess.run(self.__predict, feed_dict={self.__x: x})
			self._max_accuracy = self._sess.run(self._accuracy,{self.__x:x, self.__t:t})
			#save_path = self._saver.save(self._sess,"./save/model.ckpt")
			#print("Model save in->",save_path)
			i=0
			temp = self._wheels_num/5
			while(i < self._wheels_num):
				i += 1
				self._sess.run(tf.global_variables_initializer())
				self._sess.run(self._assign_outputW, {self.__x:x, self.__t:t})
				self._sess.run(self.__predict, feed_dict={self.__x: x})
				acc = self._sess.run(self._accuracy,{self.__x:x, self.__t:t})
				if(acc > self._max_accuracy):					self._max_accuracy = acc
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
			print("Test Accuracy: {:.9f}".format(self._sess.run(self._accuracy,{self.__x:x, self.__t:t})))
		
		
