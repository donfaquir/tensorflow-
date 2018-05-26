#coding= utf-8
import tensorflow as tf
from svd2MP import SVD 
import datetime

print('--------------------------------------------------------------------------',datetime.datetime.now())

#A = tf.constant([0,1,1,1,1,0],shape=[3,2],dtype=tf.float32)
A = tf.constant([2,4,1,3,0,0,0,0],shape=[5000,200],dtype=tf.float32)

with tf.Session() as sess:
	svd = SVD(sess,A)
	a = svd.get_MP()
	print(a.eval())
	