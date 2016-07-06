import numpy as np
import tensorflow as tf

image_size = 96
num_targets = 30

def placeholder_input():
	images_placeholder = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 1))
	targets_placeholder = tf.placeholder(tf.float32, shape=(None, num_targets))

	return images_placeholder, targets_placeholder	

def weight_variable(shape, stddev, wd=None):
	initial = tf.truncated_normal(shape, stddev=stddev)
	var = tf.Variable(initial)

	if wd is not None:
		weight_decay = tf.mul(tf.nn.l2_loss(var), wd)
		tf.add_to_collection('losses', weight_decay)

	return var

def bias_variable(shape, bd=None):
	initial = tf.constant(0.1, shape=shape)
	var = tf.Variable(initial)

	if bd is not None:
		bias_decay = tf.mul(tf.nn.l2_loss(var), bd)
		tf.add_to_collection('losses', bias_decay)

	return var

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def inference(images, keep_prob):
	# Convolutional layer 1
	with tf.name_scope('conv1'):
		kernel = weight_variable([3, 3, 1, 32], 1e-4)
		biases = bias_variable([32])

		conv1 = tf.nn.relu(conv2d(images, kernel) + biases)

	pool1 = max_pool_2x2(conv1)

	conv1_drop = tf.nn.dropout(pool1, keep_prob[0])

	# Convolutional layer 2
	with tf.name_scope('conv2'):
		kernel = weight_variable([2, 2, 32, 64], 1e-4)
		biases = bias_variable([64])

		conv2 = tf.nn.relu(conv2d(conv1_drop, kernel) + biases)

	pool2 = max_pool_2x2(conv2)

	conv2_drop = tf.nn.dropout(pool2, keep_prob[1])

	# Convolutional layer 3
	with tf.name_scope('conv3'):
		kernel = weight_variable([2, 2, 64, 128], 1e-4)
		biases = bias_variable([128])

		conv3 = tf.nn.relu(conv2d(conv2_drop, kernel) + biases)

	pool3 = max_pool_2x2(conv3)

	conv3_drop = tf.nn.dropout(pool3, keep_prob[2])

	# Fully connected layer 1
	with tf.name_scope('fc1'):
		weights = weight_variable([12 * 12 * 128, 1000], 1e-3, 5e-4)
		biases = bias_variable([1000], 5e-4)

		pool3_flat = tf.reshape(conv3_drop, [-1, 12 * 12 * 128])
		fc1 = tf.nn.relu(tf.matmul(pool3_flat, weights) + biases)

	fc1_drop = tf.nn.dropout(fc1, keep_prob[3])

	# Fully connected layer 2
	with tf.name_scope('fc2'):
		weights = weight_variable([1000, 1000], 1e-3, 5e-4)
		biases = bias_variable([1000], 5e-4)

		fc2 = tf.nn.relu(tf.matmul(fc1_drop, weights) + biases)

	fc2_drop = tf.nn.dropout(fc2, keep_prob[4])

	# Linear layer
	with tf.name_scope('linear'):
		weights = weight_variable([1000, 30], 1e-3, 5e-4)
		biases = bias_variable([30], 5e-4)

		logits = tf.matmul(fc2_drop, weights) + biases

	return logits

def loss_op(logits, targets):
	# Mean square error (MSE)
	mse_loss = tf.reduce_mean(tf.square(logits - targets))

	tf.add_to_collection('losses', mse_loss)
	loss = tf.add_n(tf.get_collection('losses'))

	return loss

def train_op(loss, learning_rate):
	optimizer = tf.train.AdamOptimizer(learning_rate)
	train = optimizer.minimize(loss)

	return train

def accuracy(predictions, targets):
	# Root mean square error (RMSE)
	accuracy = np.sqrt(np.mean(np.square(predictions - targets)))

	return accuracy

if __name__ == '__main__':
	pass
