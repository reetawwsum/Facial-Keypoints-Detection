import numpy as np
from facial_keypoints_detection_input import *
from facial_keypoints_detection_model import *
from sklearn.preprocessing import MinMaxScaler

learning_rate = 0.01
momentum = 0.9

image_scaler = MinMaxScaler(feature_range=(0, 1))
targets_scaler = MinMaxScaler(feature_range=(-1, 1))

def scaling_dataset(raw_images, raw_targets):
	images = image_scaler.fit_transform(raw_images)
	targets = targets_scaler.fit_transform(raw_targets)

	return images, targets

def unscaling_dataset(scaled_images, scaled_targets):
	images = image_scaler.inverse_transform(scaled_images)
	targets = targets_scaler.inverse_transform(scaled_targets)

	return images, targets

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

	# Training my model
	with tf.Session(graph=graph) as sess:
		# Initializing all variables
		init = tf.initialize_all_variables()
		sess.run(init)
		print 'Graph Initialized'

if __name__ == '__main__':
	run_training()
