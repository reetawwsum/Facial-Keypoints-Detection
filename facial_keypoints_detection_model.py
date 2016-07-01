import tensorflow as tf

def placeholder_input():
	images_placeholder = tf.placeholder(tf.float32, shape=(None, 9216))
	targets_placeholder = tf.placeholder(tf.float32, shape=(None, 30))

	return images_placeholder, targets_placeholder	

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def inference(images):
	# Hidden layer
	with tf.name_scope('hidden'):
		weights = weight_variable([9216, 100])
		biases = bias_variable([100])

		hidden = tf.nn.relu(tf.matmul(images, weights) + biases)

	# Linear layer
	with tf.name_scope('linear'):
		weights = weight_variable([100, 30])
		biases = bias_variable([30])

		logits = tf.matmul(hidden, weights) + biases

	return logits

def loss_op(logits, targets):
	# Mean square error (MSE)
	loss = tf.reduce_mean(tf.square(logits - targets))

	return loss

def train_op(loss, learning_rate, momentum):
	optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
	train = optimizer.minimize(loss)

	return train

def accuracy(predictions, targets):
	# Root mean square error (RMSE)
	accuracy = tf.sqrt(tf.reduce_mean(tf.square(predictions - targets)))

	return accuracy

if __name__ == '__main__':
	pass
