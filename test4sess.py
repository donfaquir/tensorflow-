'''
import tensorflow as tf

a = tf.constant([1,2], name="a", dtype= tf.float32)
b = tf.constant([2.0,3.0], name="b")
result = a + b
'''
'''
with tf.Session() as sess:
	print(sess.run(result))
'''

''' 设置默认会话
sess = tf.Session()
with sess.as_default():
	print(result.eval())
'''

''' 生成并设置默认会话
sess = tf.InteractiveSession()
print(result.eval())
sess.close()
'''
'''
# 获取labels中的值，查看文件中数据的具体组织形式
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)


import tensorflow as tf

sess = tf.InteractiveSession()

y_ = tf.placeholder(tf.float32, shape=[None, 10])
batch = mnist.train.next_batch(100)

print(y_.eval(feed_dict={y_: batch[1]}))
'''

#测试log函数的功能
import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.constant(2000,dtype=tf.float32)
print(tf.log(x).eval())