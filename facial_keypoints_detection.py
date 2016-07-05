import os
import numpy as np
from facial_keypoints_detection_input import *
from facial_keypoints_detection_model import *
from facial_keypoints_plot import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from six.moves import cPickle as pickle

learning_rate = 1e-2
max_steps = 13601
image_size = 96
batch_size = 64

file_path = 'dataset/'
train_validation_file = 'train_validation_loss.pickle'

def scaling_dataset(raw_images, raw_targets=None):
	image_scaler = MinMaxScaler(feature_range=(0, 1))
	images = image_scaler.fit_transform(raw_images)

	joblib.dump(image_scaler, 'dataset/image_scaler.pkl')

	if raw_targets is not None:
		targets_scaler = MinMaxScaler(feature_range=(-1, 1))
		targets = targets_scaler.fit_transform(raw_targets)

		joblib.dump(targets_scaler, 'dataset/targets_scaler.pkl')
		
		images = np.reshape(images, (-1, image_size, image_size, 1))
		return images, targets

	images = np.reshape(images, (-1, image_size, image_size, 1))
	return images

def unscaling_dataset(scaled_targets, scaled_images=None):
	targets_scaler = joblib.load('dataset/targets_scaler.pkl')
	targets = targets_scaler.inverse_transform(scaled_targets)

	if scaled_images is not None:
		image_scaler = joblib.load('dataset/image_scaler.pkl')
		images = image_scaler.inverse_transform(scaled_images)

		return images, targets

	return targets

def fetch_next_batch(train_images, train_targets, step):
	offset = (step * batch_size) % (len(train_images) - batch_size)

	batch_train_images = train_images[offset:(offset + batch_size)]
	batch_train_targets = train_targets[offset:(offset + batch_size)]

	indices = np.random.choice(len(batch_train_images), len(batch_train_images), replace=False)

	# Horizontal flipping the images
	batch_train_images[indices] = batch_train_images[indices, :, ::-1, :]

	# Horizontal flipping the targets
	batch_train_targets[indices, ::2] = batch_train_targets[indices, ::2] * -1 + 95

	flip_indices = [(0, 2), (1, 3), (4, 8), (5, 9), (6, 10), (7, 11), (12, 16), (13, 17), (14, 18), (15, 19), (22, 24), (23, 25)]

	for a, b in flip_indices:
		batch_train_targets[indices, a], batch_train_targets[indices, b] = batch_train_targets[indices, b], batch_train_targets[indices, a]

	return batch_train_images, batch_train_targets

def run_training():
	# Building my graph
	graph = tf.Graph()	

	with graph.as_default():
		# Creating placeholder for images and targets
		images_placeholder, targets_placeholder = placeholder_input()

		# Builds a graph that computes inference
		logits = inference(images_placeholder)

		# Adding loss op to the graph
		loss = loss_op(logits, targets_placeholder)

		# Adding train op to the graph
		train = train_op(loss, learning_rate)

		# Adding accuracy op to the graph
		score = accuracy(logits, targets_placeholder)

		# Creating saver to write training checkpoints
		saver = tf.train.Saver()

	# Training my model
	with tf.Session(graph=graph) as sess:
		# Initializing all variables
		init = tf.initialize_all_variables()
		sess.run(init)
		print 'Graph Initialized'

		print 'Loading Images'
		dataset = load_images()
		print 'Loading Complete'
		train_dataset = dataset['train_dataset']
		validation_dataset = dataset['validation_dataset']

		# Scaling the dataset
		train_images, train_targets = scaling_dataset(train_dataset.images, train_dataset.targets)
		validation_images, validation_targets = scaling_dataset(validation_dataset.images, validation_dataset.targets)

		validation_feed_dict = {images_placeholder: validation_images, targets_placeholder: validation_targets}

		train_loss = []
		validation_loss = []

		for step in xrange(max_steps):
			batch_train_images, batch_train_targets = fetch_next_batch(train_images, train_targets, step)

			feed_dict = {images_placeholder: batch_train_images, targets_placeholder: batch_train_targets}

			l, s, _ = sess.run([loss, score, train], feed_dict=feed_dict)
			train_loss.append(l)

			l1 = sess.run(loss, feed_dict=validation_feed_dict)
			validation_loss.append(l1)

			if not step % 34:
				saver.save(sess, 'dataset/my-model', global_step=step/34) 

				print 'Loss at Epoch %d: %f' % (step/34, l)
				print '  Training Accuracy: %.3f' % s
				print '  Validation Accuracy: %.3f' % sess.run(score, feed_dict=validation_feed_dict)

	# Storing train loss and validation loss in a file
	train_validation_loss = {'train_loss': np.array(train_loss), 'validation_loss': np.array(validation_loss)}

	with open(file_path + train_validation_file, 'wb') as f:
		pickle.dump(train_validation_loss, f)

def make_predictions():
	# Building my graph
	graph = tf.Graph()

	with graph.as_default():
		# Creating placeholder for images
		images_placeholder, _ = placeholder_input()

		# Building a graph inference
		logits = inference(images_placeholder)

		# Creating saver to read training checkpoints
		saver = tf.train.Saver()

	# Making prediction using saved model
	with tf.Session(graph=graph) as sess:
		saver.restore(sess, 'dataset/my-model-30')
		print 'Model Restored'

		print 'Loading dataset'
		test = read_test_file()
		print 'Dataset loaded'

		# Scaling the test dataset
		test_images = scaling_dataset(test.images)
		predictions = []

		for i, image in enumerate(test_images):
			scaled_prediction = sess.run(logits, feed_dict={images_placeholder: np.reshape(image, (1, image_size, image_size, 1))})
			prediction = unscaling_dataset(scaled_prediction)

			predictions.append(prediction[0])

		print 'Writing predictions to a file'
		write_output(predictions)			

if __name__ == '__main__':
	run_training()
	make_predictions()
