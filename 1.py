#coding= utf-8
import tensorflow as tf
from svd2MP import SVD 
import datetime

print()
print('--------------------------------------------------------------------------',datetime.datetime.now())

A = tf.constant([0,1,1,1,1,0],shape=[3,2],dtype=tf.float32)
#A = tf.constant([2,4,1,3,0,0],shape=[4,2],dtype=tf.float32)
'''
#The transposed of H
A_T = tf.transpose(A)
#Matrix transposed multiplied by matrix A_T*A
A_TA = tf.matmul(A_T, A)
#Matrix multiplied by matrix transposing A*A_T
AA_T = tf.matmul(A,A_T)

#Eigenvalues and eigenvectors of A_T*A
e_ATA,v_ATA = tf.self_adjoint_eig(A_TA)
e_AAT,v_AAT = tf.self_adjoint_eig(AA_T)
'''
with tf.Session() as sess:
	svd = SVD(sess,A)
	a = svd.get_MP()
	print(a)
	
	
	'''
	s,u,v = svd.get_svd()
	sess.run(tf.global_variables_initializer())

	
	
	s = 1.0/s
	U_T = tf.transpose(u)
	SU_T = tf.matmul(s,U_T)
	V_T = tf.transpose(v)
	VSU_T = tf.matmul(V_T,SU_T)

	for i in range(0,len(s.eval())):
		b = s[i]>1e-15*1.7
		if not b.eval():
			s[i] = 0.0
	s = 1.0/s
	diagS = tf.diag(s)
	
	m_,n_ = A.get_shape()
	#Singular value matrices for finding generalized inverse
	E = tf.Variable(tf.zeros([n_,m_]))
	
	if(sess.run(tf.greater(m_,n_))):
		m =sess.run(tf.subtract(m_,n_))
		n = n_
		zero = tf.Variable(tf.zeros([n,m]))
		E = tf.concat([diagS,zero],1)
	elif(sess.run(tf.less(m_,n_))):
		m = m_
		n = sess.run(tf.subtract(n_,m_))
		zero = tf.Variable(tf.zeros([n,m]))
		E = tf.concat([diagS,zero],0)
	else:
			E = s
	sess.run(tf.global_variables_initializer())
	
	c = tf.matmul(E,tf.transpose(u))
	print('c = E*tf.transpose(u)->',end='')
	print(c.eval())
	print('c.shape',end='')
	print(c.get_shape())

	
	
	print('E ->',end='')
	print(E.eval())
	print('E.shape->',end='')
	print(E.get_shape())
	d = tf.matmul(v,c)
	print('d = v * c',end='')
	print(d.eval())

	

	print('u.eval()',end='')
	print(u.eval())
	print('v.eval()',end='')
	print(v.eval())

	print('u.eval()',end='')
	print(u.eval())
	print('v.eval()',end='')
	print(v.eval())
	print('s.eval-->',end='')
	print(s.eval())
	
	print('v_ATA->',end='')
	print(v_ATA.eval())
	print('v_AAT->',end='')
	print(v_AAT.eval())

	
	MP = svd.get_MP()
	print(MP.eval())

	
	c = tf.matmul(s,tf.transpose(u))
	print('c = s*tf.transpose(u)->',end='')
	print(c.eval())
	print('c.shape',end='')
	print(c.get_shape())
	

	
	c = tf.matmul(E,tf.transpose(u))
	print('c = E*tf.transpose(u)->',end='')
	print(c.eval())
	print('c.shape',end='')
	print(c.get_shape())


		#The transposed of H
	A_T = tf.transpose(A)
	#Matrix transposed multiplied by matrix A_T*A
	A_TA = tf.matmul(A_T, A)
	#Matrix multiplied by matrix transposing A*A_T
	AA_T = tf.matmul(A,A_T)
	
	#Eigenvalues and eigenvectors of A_T*A
	e_ATA,v_ATA = tf.self_adjoint_eig(A_TA)
	e_AAT,v_AAT = tf.self_adjoint_eig(AA_T)
	
	
	print('diagS = tf.diag(s)->',end='')
	print(diagS.eval())
	print('diagS.shape->',end='')
	print(diagS.get_shape())
	m,n = A.get_shape()
	print('m,n = A.get_shape()->',end='')
	print(m,end='')
	print(n)

	
print('s.eval()',end='')
	print(s.eval())
	print('s.shap',end='')
	print(s.get_shape())
	

	m,n = tf.transpose(u).get_shape()

	print('m = ')
	print(m)
	print('n = ')
	print(n)
	#b = A = tf.constant([0,1,1,1,1,0],shape=[2,3],dtype=tf.float32)
	#a = s*b

	print(s.eval())
	print(s.get_shape())
	print(len(s.get_shape()))
	print(len(s.eval()))
	print(u.eval())
	print(v.eval())
	print(svd.get_cutoff(s))


	print(sess.run(v_ATA))
	print(sess.run(v_AAT))
	print(sess.run(e_ATA))
	print(sess.run(e_AAT))
				
	if('__name__' == '__main__'):  '''

