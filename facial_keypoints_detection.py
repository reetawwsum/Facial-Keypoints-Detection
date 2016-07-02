import os
import numpy as np
from facial_keypoints_detection_input import *
from facial_keypoints_detection_model import *
from facial_keypoints_plot import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

learning_rate = 0.01
momentum = 0.9
max_epochs = 401

def scaling_dataset(raw_images, raw_targets=None):
	image_scaler = MinMaxScaler(feature_range=(0, 1))
	images = image_scaler.fit_transform(raw_images)

	joblib.dump(image_scaler, 'dataset/image_scaler.pkl')

	if raw_targets is not None:
		targets_scaler = MinMaxScaler(feature_range=(-1, 1))
		targets = targets_scaler.fit_transform(raw_targets)

		joblib.dump(targets_scaler, 'dataset/targets_scaler.pkl')
		
		return images, targets

	return images

def unscaling_dataset(scaled_targets, scaled_images=None):
	targets_scaler = joblib.load('dataset/targets_scaler.pkl')
	targets = targets_scaler.inverse_transform(scaled_targets)

	if scaled_images is not None:
		image_scaler = joblib.load('dataset/image_scaler.pkl')
		images = image_scaler.inverse_transform(scaled_images)

		return images, targets

	return targets

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
		train = train_op(loss, learning_rate, momentum)

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

		train_feed_dict = {images_placeholder: train_images, targets_placeholder: train_targets}
		validation_feed_dict = {images_placeholder: validation_images, targets_placeholder: validation_targets}

		for step in xrange(max_epochs):
			l, s, _ = sess.run([loss, score, train], feed_dict=train_feed_dict)

			if not step % 50:
				saver.save(sess, 'dataset/my-model', global_step=step) 

				print 'Loss at Epoch %d: %f' % (step, l)
				print '  Training Accuracy: %.3f' % s
				print '  Validation Accuracy: %.3f' % sess.run(score, feed_dict=validation_feed_dict)

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
		saver.restore(sess, 'dataset/my-model-400')
		print 'Model Restored'

		print 'Loading dataset'
		test = read_test_file()
		print 'Dataset loaded'

		# Scaling the test dataset
		test_images = scaling_dataset(test.images)
		predictions = []

		for i, image in enumerate(test_images):
			scaled_prediction = sess.run(logits, feed_dict={images_placeholder: np.reshape(image, (1, 9216))})
			prediction = unscaling_dataset(scaled_prediction)

			predictions.append(prediction[0])

		print 'Writing predictions to a file'
		write_output(predictions)			

if __name__ == '__main__':
	run_training()
	make_predictions()
