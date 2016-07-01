import numpy as np
from facial_keypoints_detection_input import *
from facial_keypoints_detection_model import *
from sklearn.preprocessing import MinMaxScaler

learning_rate = 0.01
momentum = 0.9

def preprocessing_dataset(raw_images, raw_targets):
	images = MinMaxScaler(feature_range=(0, 1)).fit_transform(raw_images)
	targets = MinMaxScaler(feature_range=(-1, 1)).fit_transform(raw_targets)

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

	# Training my model
	with tf.Session(graph=graph) as sess:
		# Initializing all variables
		init = tf.initialize_all_variables()
		sess.run(init)
		print 'Graph Initialized'

if __name__ == '__main__':
	run_training()
