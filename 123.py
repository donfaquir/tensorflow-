import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE = 500

BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 3000
MOVING_AVERAGE_DECAY = 0.99

def inference(input_tensor,avg_class,weight1,biases1,weight2,biases2):
	if avg_class == None:
		layer1 = tf.nn.relu(tf.matmul(input_tensor,weight1)+biases1)
		return tf.matmul(layer1,weight2) + biases2
	else:
		layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weight1)) + 
			avg_class.average(biases1))
		return tf.matmul(layer1,avg_class.average(weight2)) + avg_class.average(biases2)
def train(mnist):
	x = tf.placeholder(tf.float32,[None,INPUT_NODE],name = 'x-input')
	y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')
	
	weight1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE], stddev = 0.1))
	biases1 = tf.Variable(tf.constant(0.1,shape = [LAYER1_NODE]))
	
	weight2 = tf.Variable( tf.truncated_normal( [LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
	biases2 = tf.Variable( tf.constant(0.1,shape=[OUTPUT_NODE]))
	
	y = inference(x,None,weight1,biases1,weight2,biases2)
	
	global_step = tf.Variable(0, trainable=False)
	
	variable_average = tf.train.ExponentialMovingAverage( MOVING_AVERAGE_DECAY,global_step)
	
	variable_average_op = variable_average.apply( tf.trainable_variables())
	
	average_y = inference(x,variable_average,weight1,biases1,weight2,biases2)
	
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	
	regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
	regularization = regularizer(weight1) + regularizer(weight2)
	
	loss = cross_entropy_mean + regularization
	
	learing_rete = tf.train.exponential_decay( LEARNING_RATE_BASE, global_step,
		mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY )
	
	train_step = tf.train.GradientDescentOptimizer(learing_rete).minimize(loss,	global_step = global_step)

	with tf.control_dependencies([train_step,variable_average_op]):
		train_op = tf.no_op(name='train')

	correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

	with tf.Session() as sess:
		init_op=tf.global_variables_initializer()
		sess.run(init_op)
		
		validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
		
		test_feed = {x:mnist.test.images,y_:mnist.test.labels}
		
		for i in range(TRAINING_STEPS):
			if i%100 == 0:
				validate_acc = sess.run(accuracy,feed_dict = validate_feed)
				print("After %d trainint step(s),validation accuracy"
					"using average model is %g" % (i,validate_acc))
			xs,ys = mnist.train.next_batch(BATCH_SIZE)
			sess.run(train_op,feed_dict={x:xs,y_:ys})
		test_acc = sess.run(accuracy,feed_dict=test_feed)
		print("After %d trainint step(s),validation accuracy"
			"using average model is %g" % (TRAINING_STEPS,test_acc))

def main(argv=None):
	mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
	train(mnist)
	
if __name__ == '__main__':
	tf.app.run()
	
	
		
		
		
		
	