import tensorflow as tf

image_size = 96
num_targets = 30

def placeholder_input():
	images_placeholder = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 1))
	targets_placeholder = tf.placeholder(tf.float32, shape=(None, num_targets))

	return images_placeholder, targets_placeholder	

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def inference(images):
	# Convolutional layer 1
	with tf.name_scope('conv1'):
		weights = weight_variable([3, 3, 1, 32])
		biases = bias_variable([32])

		conv1 = tf.nn.relu(conv2d(images, weights) + biases)

	pool1 = max_pool_2x2(conv1)

	conv1_drop = tf.nn.dropout(pool1, 0.1)

	# Convolutional layer 2
	with tf.name_scope('conv2'):
		weights = weight_variable([2, 2, 32, 64])
		biases = bias_variable([64])

		conv2 = tf.nn.relu(conv2d(conv1_drop, weights) + biases)

	pool2 = max_pool_2x2(conv2)

	conv2_drop = tf.nn.dropout(pool2, 0.2)

	# Convolutional layer 3
	with tf.name_scope('conv3'):
		weights = weight_variable([2, 2, 64, 128])
		biases = bias_variable([128])

		conv3 = tf.nn.relu(conv2d(conv2_drop, weights) + biases)

	pool3 = max_pool_2x2(conv3)

	conv3_drop = tf.nn.dropout(pool3, 0.3)

	# Fully connected layer 1
	with tf.name_scope('fc1'):
		weights = weight_variable([12 * 12 * 128, 1000])
		biases = bias_variable([1000])

		pool3_flat = tf.reshape(conv3_drop, [-1, 12 * 12 * 128])
		fc1 = tf.nn.relu(tf.matmul(pool3_flat, weights) + biases)

	fc1_drop = tf.nn.dropout(fc1, 0.5)

	# Fully connected layer 2
	with tf.name_scope('fc2'):
		weights = weight_variable([1000, 1000])
		biases = bias_variable([1000])

		fc2 = tf.nn.relu(tf.matmul(fc1_drop, weights) + biases)

	# Linear layer
	with tf.name_scope('linear'):
		weights = weight_variable([1000, 30])
		biases = bias_variable([30])

		logits = tf.matmul(fc2, weights) + biases

	return logits

def loss_op(logits, targets):
	# Mean square error (MSE)
	loss = tf.reduce_mean(tf.square(logits - targets))

	return loss

def train_op(loss, learning_rate):
	optimizer = tf.train.AdamOptimizer(learning_rate)
	train = optimizer.minimize(loss)

	return train

def accuracy(predictions, targets):
	# Root mean square error (RMSE)
	accuracy = tf.sqrt(tf.reduce_mean(tf.square(predictions - targets)))

	return accuracy

if __name__ == '__main__':
	pass
