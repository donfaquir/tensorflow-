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
		
		assert isinstance(H, tf.Tensor), "Number of batch_size must be integer" 
		self.__A = H
		self.__sess = sess;
		#Obtaining the dimension information of the matrix
		self.__shape = self.__A.get_shape()
		
		#Get the first dimension information of the matrix, the line number
		self.__row_num = self.__shape[0]
		#Get the second dimension information of the matrix, that is, the number of columns
		self.__columns_num = self.__shape[1]
		
		self.__svd = tf.placeholder(tf.float32, shape=[self.__row_num, self.__columns_num])
	def trans(self,H):
		'''
			左右翻转矩阵
		'''
	def get_svd(self):
		#The transposed of H
		A_T = tf.transpose(self.__A)
		#Matrix transposed multiplied by matrix A_T*A
		A_TA = tf.matmul(A_T, self.__A)
		#Matrix multiplied by matrix transposing A*A_T
		AA_T = tf.matmul(self.__A,A_T)
		
		#Eigenvalues and eigenvectors of A_T*A
		e_ATA,v_ATA = tf.self_adjoint_eig(A_TA)
		e_AAT,v_AAT = tf.self_adjoint_eig(AA_T)
		
		#The diagonal matrix with the reciprocal of the square of the value of A_T*A is diagonal value.
		h2 = tf.diag(tf.transpose(tf.sqrt(1/e_ATA)))
		h = tf.diag(tf.sqrt(1/e_ATA))
		
		#Singular value matrices for finding generalized inverse
		E = tf.Variable(tf.zeros([self.__row_num,self.__columns_num]))
		if(self.__sess.run(tf.greater(self.__row_num,self.__columns_num))):
			m =self.__sess.run(tf.subtract(self.__row_num,self.__columns_num))
			n = self.__columns_num
			zero = tf.Variable(tf.zeros([m,n]))
			E = tf.concat([h,zero],0)
		elif(self.__sess.run(tf.less(self.__row_num,self.__columns_num))):
			m = self.__row_num,dtype
			n = tf.subtract(self.__columns_num,self.__row_num)
			zero = tf.Variable(tf.zeros([m,n]))
			E = tf.concat([h,zero],1)
		else:
			E = h
		
		U_ = v_AAT
		V_ = v_ATA
		#Singular value decomposition of matrices(H = U*E*V_T)
		self.__svd = tf.matmul(tf.matmul(U_,E),tf.transpose(V_))
		#Moore-Penrose generalized inverse of a matrix（H+ = v*E*U_T）
		MP = tf.matmul(tf.matmul(V_,tf.transpose(E)),tf.transpose(U_))
		#return (self.__svd,self.__shape,self.__columns_num,self.__row_num,A_T,A_TA,AA_T,e_ATA,v_ATA,e_AAT,v_AAT)
		return (h,h2)
	def get_svd2(self):
		s,u,v = tf.svd(self.__A);
		return s,u,v
		
	@property
	def shape(self):
		return self.__shape
	@property
	def row_num(self):
		return self.__row_num
	@property
	def columns_num(self):
		return self.__columns_num