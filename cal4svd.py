#coding=utf-8
import tensorflow as tf
'''
	The Moore_Penrose Generalized Inverse of Solving Singular Value Decomposition Matrices
'''
class SVD(object):
	'''
		H:A matrix that needs to find the generalized inverse of Moore_Penrose，
		The input form must be the tensor of tensorflow, that is, this code can only be invoked in tensorflow environment.
	'''
	def __init__(self,sess,H):
		
		assert isinstance(H, tf.Tensor), "Input valuse must be Tensor" 
		self._A = H
		self._sess = sess;
		#Obtaining the dimension information of the matrix
		self._shape = self._A.get_shape()
		
		#Get the first dimension information of the matrix, the line number
		self._row_num = self._shape[0]
		#Get the second dimension information of the matrix, that is, the number of columns
		self._columns_num = self._shape[1]
		
		self._svd = tf.placeholder(tf.float32, shape=[self._row_num, self._columns_num])
	def converter(self,H):
		'''
			Columns of matrix Left and right exchange
		'''
		H1 = H.eval()
		dimensionN = len(H.get_shape())
		#Processing one dimensional matrix
		if(dimensionN == 1):
			coluN = H.get_shape()[0]
			left = 0
			right = coluN -1
			while(left < right):
				temp = H1[left]
				H1[left] = H1[right]
				H1[right] = temp
				left = left +1
				right = right -1
		#Processing two dimensional matrix
		elif(dimensionN == 2):
			rowN,coluN = H.get_shape()
			left = 0
			right = coluN -1
			while(left < right):
				for i in range(0,rowN):
					temp = H1[i][left]
					H1[i][left] = H1[i][right]
					H1[i][right] = temp
				left = left +1
				right = right -1
		else:
			raise ValueError(
				'Can only handle a matrix with a dimension of 1 or 2.'
			)
		
		return H1
	def get_svd(self):
		'''
			The solution method I realized by myself
		'''
		#The transposed of H
		A_T = tf.transpose(self._A)
		#Matrix transposed multiplied by matrix A_T*A
		A_TA = tf.matmul(A_T, self._A)
		#Matrix multiplied by matrix transposing A*A_T
		AA_T = tf.matmul(self._A,A_T)
		
		#Eigenvalues and eigenvectors of A_T*A
		e_ATA,v_ATA = tf.self_adjoint_eig(A_TA)
		e_AAT,v_AAT = tf.self_adjoint_eig(AA_T)
		
		
		#Conversion of eigenvalues from non subtraction to non increasing sequences
		b = self.converter(tf.convert_to_tensor(tf.sqrt(1/e_ATA)))
		#The diagonal matrix with the reciprocal of the square of the value of A_T*A is diagonal value.
		s = tf.diag(b)
		
		#Singular value matrices for finding generalized inverse
		E = tf.Variable(tf.zeros([self._row_num,self._columns_num]))
		if(self._sess.run(tf.greater(self._row_num,self._columns_num))):
			m =self._sess.run(tf.subtract(self._row_num,self._columns_num))
			n = self._columns_num
			zero = tf.Variable(tf.zeros([m,n]))
			E = tf.concat([s,zero],0)
		elif(self._sess.run(tf.less(self._row_num,self._columns_num))):
			m = self._row_num
			n = self._sess.run(tf.subtract(self._columns_num,self._row_num))
			zero = tf.Variable(tf.zeros([m,n]))
			E = tf.concat([s,zero],1)
		else:
			E = s
		
		U_ = self.converter(v_AAT)
		V_ = self.converter(v_ATA)
		#Singular value decomposition of matrices(H = U*E*V_T)
		#self._svd = tf.matmul(tf.matmul(U_,E),tf.transpose(V_))
		
		return (U_,E,V_,self._svd)
	
	def get_svd_MP(self):
		s,u,v = tf.svd(self._A);
		
		#Moore-Penrose generalized inverse of a matrix（H+ = v*E*U_T）
		MP = tf.matmul(tf.matmul(V_,tf.transpose(E)),tf.transpose(U_))
		return s,u,v
		
	@property
	def shape(self):
		return self._shape
	@property
	def row_num(self):
		return self._row_num
	@property
	def columns_num(self):
		return self._columns_num