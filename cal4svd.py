import tensorflow as tf
'''
	求解矩阵的Moore_Penrose广义逆
'''
class SVD(object):
	'''
		H:需要求Moore_Penrose广义逆的矩阵，
		输入形式必须是tensorflow的张量，即此代码只能在tensorflow环境下调用
	'''
	def __init__(self,H):
		assert isinstance(H, tf.Tensor), "Number of batch_size must be integer" 
		self.__A = H
		self.__sess = tf.Session()
		#获取矩阵的维度信息
		self.__shape = self.__A.get_shape()
		
		#获取矩阵的第一维度信息，即行数
		self.__row_num = self.__shape[0]
		#获取矩阵的第二维度信息，即列数
		self.__columns_num = self.__shape[1]
		
		self.__svd = tf.placeholder(tf.float32, shape=[self.__row_num, self.__columns_num])
	def get_svd(self):
		#矩阵的转置A_T
		A_T = tf.transpose(self.__A)
		#矩阵转置与矩阵乘积 A_T*A
		A_TA = tf.matmul(A_T, self.__A)
		#矩阵与矩阵转置乘积 A*A_T
		AA_T = tf.matmul(self.__A,A_T)
		
		#A_T*A的特征值和特征向量
		e_ATA,v_ATA = tf.self_adjoint_eig(A_TA)
		e_AAT,v_AAT = tf.self_adjoint_eig(AA_T)
		
		#以A_T*A特征值开方为对角线值得对角矩阵
		h = tf.diag(tf.sqrt(e_ATA))
		
		#求广义逆的奇异值矩阵
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
		
		self.__svd = tf.matmul(tf.matmul(U_,E),tf.transpose(V_))
		
		return (self.__svd,self.__shape,self.__columns_num,self.__row_num,A_T,A_TA,AA_T,e_ATA,v_ATA,e_AAT,v_AAT)
		
	@property
	def shape(self):
		return self.__shape
	@property
	def row_num(self):
		return self.__row_num
	@property
	def columns_num(self):
		return self.__columns_num