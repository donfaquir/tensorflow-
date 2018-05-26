#coding=utf-8
import tensorflow as tf
from cal4svd import SVD
'''
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
A = tf.constant([0,1,1,1,1,0],shape=[3,2],dtype=tf.float32)
assert isinstance(A, tf.Tensor), "Number of batch_size must be integer" 

m = A.get_shape();
print('m = {}'.format(m[1]))
print(A.get_shape())
print(A.get_shape()[1])


labels = [1, 2, 3]
x = tf.expand_dims(labels, 0)
print("为张量+1维，但是X执行的维度维0,则不更改", sess.run(x))
x = tf.expand_dims(labels, 1)
print("为张量+1维，X执行的维度维1,则增加一维度", sess.run(x))
x = tf.expand_dims(labels, -1)
print("为张量+1维，但是X执行的维度维-1,则不更改", sess.run(x))
'''
def main():
	A = tf.constant([0,1,1,1,1,0],shape=[3,2],dtype=tf.float32)
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		svd = SVD(sess,A)
		'''
		H = svd.converter(A)
		print(A.eval())
		print(H)
		
		
		s,u,v = H.get_svd2();
		print(s.eval())
		print(u.eval())
		print(v.eval())
		'''
		s,u,v = svd.get_svd();
		print(s.eval())
		print(u.eval())
		print(v.eval())
main()
'''
if('__name__'=='__main__'):
	main()
'''	
