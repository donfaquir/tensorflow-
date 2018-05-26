#coding = utf-8
import tensorflow as tf
'''
	The Moore_Penrose Generalized Inverse of Solving Singular Value Decomposition Matrices
	author:caichangqing
	time:2018-5-6 00:23:48
'''
class SVD(object):
	'''
		H:A matrix that needs to find the generalized inverse of Moore_Penrose，
		The input form must be the tensor of tensorflow, that is, this code can only be invoked in tensorflow environment.
	'''
	def __init__(self,sess,H):
	
		if not (isinstance(H, tf.Tensor)):
			if not isinstance(H, tf.Variable):
				raise TypeError(
						"Input valuse must be Tensor or Variable"
					)
		self._A = H
		self._sess = sess;
		#Obtaining the dimension information of the matrix
		self._shape = self._A.get_shape()
		
		#Get the first dimension information of the matrix, the line number
		self._row_num = self._shape[0]
		#Get the second dimension information of the matrix, that is, the number of columns
		self._columns_num = self._shape[1]
		
	def get_svd(self):
		return tf.svd(self._A,full_matrices=True)
	
	def get_MP(self):
		s,u,v = self.get_svd()
		'''
			The following is an explanation of the official website's return type of tf.svd() function
			<--- tensor[..., :, :] = u[..., :, :] * diag(s[..., :, :]) * transpose(conj(v[..., :, :]))    ---->
			We can know that: get_MP() ==> transpose(v)*1.0/s*transpose(u)
			You can verify the correctness of get_MP() with the pinv() function of numpy library
		'''
		s = 1.0/s
		diagS = tf.diag(s)
		
		#Singular value matrices for finding generalized inverse(N*M)
		E = tf.Variable(tf.zeros([self._columns_num,self._row_num]))
		#Extended eigenvalue matrix,So that its dimension conforms to the requirement of multiplication with U matrix.
		if(self._sess.run(tf.greater(self._row_num,self._columns_num))):
			m =self._sess.run(tf.subtract(self._row_num,self._columns_num))
			n = self._columns_num
			zero = tf.Variable(tf.zeros([n,m]))
			E = tf.concat([diagS,zero],1)
		elif(self._sess.run(tf.less(self._row_num,self._columns_num))):
			m = self._row_num
			n = self._sess.run(tf.subtract(self._columns_num,self._row_num))
			zero = tf.Variable(tf.zeros([n,m]))
			E = tf.concat([diagS,zero],0)
		else:
			E = diagS
	
		self._sess.run(tf.global_variables_initializer())
		
		U_T = tf.transpose(u)
		SU_T = tf.matmul(E,U_T)
		V_T = tf.transpose(v)
		VSU_T = tf.matmul(V_T,SU_T)
		
		return VSU_T