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
		H = SVD(sess,A);
		s,u,v = H.get_svd2();
		print(s.eval())
		print(u.eval())
		print(v.eval())
		#(svd,shapexx,columns_num,row_num,A_T,A_TA,AA_T,e_ATA,v_ATA,e_AAT,v_AAT) = H.get_svd()
	#h,h2 = H.get_svd2()
	#sess = tf.InteractiveSession()
		'''print('svd = {}')
		print(svd.eval())
		
		print('shape = {}')
		print(shapexx)
		print('columns_num = {}')
		print(columns_num)
		
		print('row_num = {}')
		print(row_num)
		print('A_T = {}')
		print(A_T.eval())
		print('A_TA = {}')
		print(A_TA.eval())
		print('AA_T = {}')
		print(AA_T.eval())
		print('e_ATA = {}')
		print(e_ATA.eval())
		print('v_ATA = {}')
		print(v_ATA.eval())
		print('e_AAT = {}')
		print(e_AAT.eval())
		print('v_AAT = {}')
		print(v_AAT.eval())
		
		print('h = ')
		print(h.eval())
		print('h2 = ')
		print(h2.eval())'''
	
main()
'''
if('__name__'=='__main__'):
	main()
'''	
