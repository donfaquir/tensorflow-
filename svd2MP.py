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
		
		assert isinstance(H, tf.Tensor), "Input valuse must be Tensor" 
		self._A = H
		#self._sess = sess;
		self._sess = tf.InteractiveSession()
		#sess.tf.InteractiveSession()
		#Obtaining the dimension information of the matrix
		self._shape = self._A.get_shape()
		
		#Get the first dimension information of the matrix, the line number
		self._row_num = self._shape[0]
		#Get the second dimension information of the matrix, that is, the number of columns
		self._columns_num = self._shape[1]
		
		self._svd = tf.placeholder(tf.float32, shape=[self._row_num, self._columns_num])
		
	def get_svd(self):
		return tf.svd(self._A,full_matrices=True)
	
	def get_MP(self):
		s,u,v = self.get_svd()
		
		#Get the inverse of the singular value matrix
		cutoff = self.get_cutoff(self._A)
		for i in range(0,len(s.eval())):
			b = s[i]>cutoff
			if not b.eval():
				s[i] = 0.0
		s = 1.0/s
		diagS = tf.diag(s)
		
		self._row_num,self._columns_num = self._A.get_shape()
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
			E = s
		self._sess.run(tf.global_variables_initializer())
		U_T = tf.transpose(u)
		SU_T = tf.matmul(E,U_T)
		VSU_T = tf.matmul(v,SU_T)
		
		return VSU_T
		
	def get_cutoff(self,H):
		#va = self._sess(H.eval())
		#return 1e-15*va.max()
		return 1e-10*1.0