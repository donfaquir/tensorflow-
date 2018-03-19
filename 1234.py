import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST数据集相关的常用
INPUT_NODE=784
OUTPUT_NODE=10

# 配置神经网络的参数
LAYER1_NODE=500 # 隐藏层节点个数
BATCH_SIZE=100 # 一个训练batch中的数据个数。
LEARNING_RATE_BASE=0.8 # 基础的学习率
LEARNING_RATE_DECAY=0.99 # 学习率的衰减率
REGULARIZATION_RATE=0.0001 # 描述模型复杂度的正则化项在损失函数中的系统
TRAINING_STEPS=30000 # 训练轮数
MOVING_AVERAGE_DECAY=0.99 # 滑动平均衰减率

# 定义一个三层神经网络。
def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    # 当没有提供滑动平均类时，直接使用参数当前的取值。
    if avg_class==None:
        # 计算隐藏层的前向传播结果，这里使用ReLU激活函数
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        return tf.matmul(layer1,weights2)+biases2
    else:
        # 首先使用avg_class.average函数来计算出变量的滑动平均值，
        # 然后再计算相应的神经网络前向传播结果
        layer1=tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))+
                          avg_class.average(biases1))
        return (tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2))
    
    
# 训练模型的过程
def train(mnist):
    x=tf.placeholder(tf.float32,shape=[None,INPUT_NODE],name='x-input')
    y_=tf.placeholder(tf.float32,shape=[None,OUTPUT_NODE],name='y-input')
    
    # 生成隐藏层的参数
    weights1=tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
    # 生成输出层的参数
    weights2=tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
    
    # 计算在当前参数下神经网络前向传播结果。这里给出的用于计算滑动平均的类为None，
    # 所以函数不会使用参数的滑动平均值。
    y=inference(x,None,weights1,biases1,weights2,biases2)
    
    # 定义存储训练轮数的变量。该变量为不可训练变量(trainable=False)
    global_step=tf.Variable(0,trainable=False)
    
    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类。给定训练轮数的变量
    # 可以加快训练早期变量的更新速度。
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    
    # 在所有代表神经网络参数的变量上使用滑动平均。tf.trainable_variables返回的就是图上集合
    # GraphKeys.TRAINABLE_VARIABLES中的元素。这个集合的元素就是所有没有指定
    # trainable=False的参数。
    variables_averages_op=variable_averages.apply(tf.trainable_variables())
    
    # 计算使用了滑动平均之后的前向传播结果。滑动平均不会改变变量本身的值，而是维护一个影子
    # 变量来记录其滑动平均值。所以当需要使用这个滑动平均值时，需要明确调用average函数
    average_y=inference(x,variable_averages,weights1,biases1,weights2,biases2)
    
    # 使用交叉熵作为损失函数。这里使用
    # sparse_softmax_cross_entropy_with_logits函数来计算交叉熵。因为标准答案是一个长度为
    # 10的一维数组，而该函数需要提供的是一个正确答案的数字，所以需要使用tf.argmax函数来
    # 得到正确答案对应的类别编号
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    
    # 计算L2正则化损失函数
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失。一般只计算神经网络边上权重的正则化损失，而不使用偏置项
    regularization=regularizer(weights1)+regularizer(weights2)
    # 总损失等于交叉熵损失和正则化损失的和
    loss=cross_entropy_mean+regularization
    # 设置指数衰减的学习率。
    learning_rate=tf.train.exponential_decay(
        LEARNING_RATE_BASE, # 基础的学习率，随着迭代的进行，更新变量时使用的学习率
                                # 在这个基础上递减
        global_step, # 当前迭代的轮数
        mnist.train.num_examples/BATCH_SIZE, # 过完所有的训练数据需要的迭代次数
        LEARNING_RATE_DECAY # 学习率衰减速度
    )
    
    # 使用tf.train.GradientDescentOptimizer优化算法来优化损失函数。这里的损失函数
    # 包含了交叉熵损失和L2正则化损失。
    train_step=tf.train.GradientDescentOptimizer(learning_rate)\
                .minimize(loss,global_step=global_step)
    
    # 在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新网络中的参数，
    # 又要更新每一个参数的滑动平均值。为了一次完成多个操作，tensorflow提供了
    # tf.control_dependencies和tf.group两种机制。下面两行程序和
    # train_op=tf.group(train_step,variables_averages_op)是等价的。
    with tf.control_dependencies([train_step,variables_averages_op]):
        train_op=tf.no_op(name='train')
        
    # 检验使用了滑动平均模型的神经网络前向传播结果是否正确。tf.argmax(average_y,1)
    # 计算每一个样例的预测答案。其中average_y是一个batch_size*10的二维数组，每一行
    # 表示一个样例的前向传播结果。tf.argmax的第二个参数‘1’表示选取最大值的操作仅在
    # 第一个维度中进行，也就是说，只在每一行选取最大值对应的下标。于是得到的结果是
    # 一个长度为batch的以为数组，这个一维数组中的值就表示了每一个样例对应的数字识别结果。
    # tf.equal判断两个张量的每一维是否相等。
    correct_prediction=tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    # 初始化会话并开始训练过程。
    with tf.Session() as sess:
        init_op=tf.global_variables_initializer()
        sess.run(init_op)
        # 准备验证数据。一般在神经网络的训练过程中会通过验证数据来大致判断停止的
        # 条件和评判训练的效果。
        validate_feed={x:mnist.validation.images,y_:mnist.validation.labels}
        
        # 准备测试数据
        test_feed={x:mnist.test.images,y_:mnist.test.labels}
        
        # 迭代的训练神经网络
        for i in range(TRAINING_STEPS):
            # 每1000轮输出一次在验证数据集上的测试结果
            if i%1000==0:
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                print('After {} training step(s),validation accuracy'
                      'using average model is {}'.format(i,validate_acc))
                
            # 产生这一轮使用的一个batch的训练数据，并运行训练过程。
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
        
        # 在训练结束后，在测试数据上检测神经网络模型的最终正确率
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print('After {} training step(s), test accuracy using average '
             'model is {}'.format(TRAINING_STEPS,test_acc))

def main(argv=None):
    # 声明处理MNIST数据集的类，这个类在初始化时会自动下载数据
    mnist=input_data.read_data_sets('.',one_hot=True)
    train(mnist)

# tensorflow 提供的一个主程序入口，tf.app.run会调用上面定义的main函数
if __name__=='__main__':
    tf.app.run()