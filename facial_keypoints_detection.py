import os
import numpy as np
from facial_keypoints_detection_input import *
from facial_keypoints_detection_model import *
from facial_keypoints_plot import *

learning_rate = 1e-3
max_steps = 13601
image_size = 96
batch_size = 64

def run_training():
	# Building my graph
	graph = tf.Graph()	

	with graph.as_default():
		# Creating placeholder for images and targets
		images_placeholder, targets_placeholder = placeholder_input()

		# Creating placeholder for dropout
		keep_prob = tf.placeholder(tf.float32, shape=(5))

		# Builds a graph that computes inference
		logits = inference(images_placeholder, keep_prob)

		# Adding loss op to the graph
		loss = loss_op(logits, targets_placeholder)

		# Adding train op to the graph
		train = train_op(loss, learning_rate)

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

		validation_feed_dict = {images_placeholder: validation_images, targets_placeholder: validation_targets, keep_prob: [1.0, 1.0, 1.0, 1.0, 1.0]}

		unscaled_validation_targets = unscaling_dataset(validation_targets)

		for step in xrange(max_steps):
			batch_train_images, batch_train_targets = fetch_next_batch(train_images, train_targets, step)

			feed_dict = {images_placeholder: batch_train_images, targets_placeholder: batch_train_targets, keep_prob: [0.9, 0.8, 0.7, 0.5, 0.5]}

			scaled_train_predictions, l, _ = sess.run([logits, loss, train], feed_dict=feed_dict)

			if not step % 34:
				saver.save(sess, 'dataset/my-model', global_step=step/34) 

				unscaled_train_predictions = unscaling_dataset(scaled_train_predictions)
				unscaled_train_targets = unscaling_dataset(batch_train_targets)

				scaled_validation_predictions, l1 = sess.run([logits, loss], feed_dict=validation_feed_dict)

				unscaled_validation_predictions = unscaling_dataset(scaled_validation_predictions)

				# Logging training and validation loss
				log(l, l1)

				print 'Loss at Epoch %d: %f' % (step/34, l)
				print '  Training Score: %.3f' % accuracy(unscaled_train_predictions, unscaled_train_targets)
				print '  Validation Score: %.3f' % accuracy(unscaled_validation_predictions, unscaled_validation_targets)

def make_predictions():
	# Building my graph
	graph = tf.Graph()

	with graph.as_default():
		# Creating placeholder for images
		images_placeholder, _ = placeholder_input()

		# Creating placeholder for dropout
		keep_prob = tf.placeholder(tf.float32, shape=(5))

		# Building a graph inference
		logits = inference(images_placeholder, keep_prob)

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
			scaled_prediction = sess.run(logits, feed_dict={images_placeholder: np.reshape(image, (1, image_size, image_size, 1)), keep_prob: [1.0, 1.0, 1.0, 1.0, 1.0]})
			prediction = unscaling_dataset(scaled_prediction)

			predictions.append(prediction[0])

		print 'Writing predictions to a file'
		write_output(predictions)			

if __name__ == '__main__':
	run_training()
	make_predictions()
