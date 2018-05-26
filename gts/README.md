### 训练交通标志数据集 ###
[参考文章](https://blog.csdn.net/sinat_34686158/article/details/77719436)

#### tfELM.py ####
基于一种正则化的正交投影求解ELM的输出权值，
#### train_tf.py ####
使用数据集测试实现的有效性
#### tfmpELM.py ####
输出数据矩阵与其转置的乘积的逆，必定是一个方阵，MP逆就可以用矩阵逆替换。
HB=T => (H*H_T)B = H_T*T => B = (H*H_T)-1*H_T*T

在train_tfmp.py中，特意对十张测试图片进行测试，并把结果可视化
> 2018-5-26 22:41:03