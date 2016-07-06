import csv
import numpy as np
from datetime import datetime
from six.moves import cPickle as pickle
from sklearn.externals import joblib
from sklearn import cross_validation
from sklearn.preprocessing import MinMaxScaler

file_path = 'dataset/'
train_file = 'training.csv'
test_file = 'test.csv'
log_file = 'log.csv'
image_size = 96
batch_size = 64

class facial:
	pass

keypoints = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'left_eye_inner_corner_x', 'left_eye_inner_corner_y', 'left_eye_outer_corner_x', 'left_eye_outer_corner_y', 'right_eye_inner_corner_x', 'right_eye_inner_corner_y', 'right_eye_outer_corner_x', 'right_eye_outer_corner_y', 'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y', 'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y', 'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y', 'nose_tip_x', 'nose_tip_y', 'mouth_left_corner_x', 'mouth_left_corner_y', 'mouth_right_corner_x', 'mouth_right_corner_y', 'mouth_center_top_lip_x', 'mouth_center_top_lip_y', 'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y']

def read_train_file(max_images=None):
	with open(file_path + train_file, 'rb') as f:
		csv_reader = csv.reader(f, delimiter=',')

		csv_reader.next()
		targets = []
		images = []

		for row in csv_reader:
			if len(images) == max_images:
				break

			target = row[:-1]

			if not np.sum(np.array(target) == ''):
				targets.append(target)
				images.append(row[-1].split())

		images = np.array(images).astype(float)
		targets = np.array(targets).astype(float)

		train = facial()
		train.images = images
		train.targets = targets

	return train

def read_test_file(max_images=None):
	with open(file_path + test_file, 'rb') as f:
		csv_reader = csv.reader(f, delimiter=',')

		csv_reader.next()
		images = []

		for row in csv_reader:
			if len(images) == max_images:
				break

			images.append(row[-1].split())
		
		images = np.array(images).astype(float)

		test = facial()
		test.images = images

	return test

def write_output(content, file_name='output.csv'):
	with open(file_path + file_name, 'wb') as f:
		csv_writer = csv.writer(f, delimiter=',')
		csv_writer.writerow(['RowId', 'Location'])	

		with open(file_path + 'IdLookupTable.csv') as g:
			csv_reader = csv.reader(g, delimiter=',')
			csv_reader.next()

			for row in csv_reader:
				row_id = row[0]
				target = content[int(row[1])-1]
				keypoint_index = keypoints.index(row[2])
				keypoint_location = target[keypoint_index]

				csv_writer.writerow([row_id, keypoint_location])

def split_train_dataset(train_dataset, validation_size):
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_dataset.images, train_dataset.targets, test_size=validation_size, random_state=42)

	train = facial()
	validation = facial()

	train.images = X_train
	train.targets = y_train
	validation.images = X_test
	validation.targets = y_test

	return train, validation

def load_images(validation_size=0.1):
	train_dataset = read_train_file()
	train, validation = split_train_dataset(train_dataset, validation_size)

	dataset = {'train_dataset': train, 'validation_dataset': validation}

	return dataset

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

	# Horizontal flipping the images
	horizontal_flipped_images = batch_train_images[:, :, ::-1, :]
	batch_train_images = np.append(batch_train_images, horizontal_flipped_images, axis=0)

	# Horizontal flipping the targets
	batch_train_targets = np.append(batch_train_targets, batch_train_targets, axis=0)
	batch_train_targets[batch_size:, ::2] = batch_train_targets[batch_size:, ::2] * -1

	flip_indices = [(0, 2), (1, 3), (4, 8), (5, 9), (6, 10), (7, 11), (12, 16), (13, 17), (14, 18), (15, 19), (22, 24), (23, 25)]

	for a, b in flip_indices:
		batch_train_targets[batch_size:, [a, b]] = batch_train_targets[batch_size:, [b, a]]

	return batch_train_images, batch_train_targets

def log(train_loss, validation_loss):
	with open(file_path + log_file, 'ab') as f:
		csv_writer = csv.writer(f, delimiter=',')
		csv_writer.writerow([train_loss, validation_loss, datetime.now().strftime('%H:%M:%S %d-%m-%Y')])

def load_train_validation_loss():
	with open(file_path + log_file, 'rb') as f:
		csv_reader = csv.reader(f, delimiter=',')

		train_loss = []
		validation_loss = []
		timestamp = []

		for row in csv_reader:
			train_loss.append(row[0])
			validation_loss.append(row[1])
			timestamp.append(row[2])

	train_validation_loss = {'train_loss': train_loss, 'validation_loss': validation_loss, 'timestamp': timestamp}

	return train_validation_loss

if __name__ == '__main__':
	dataset = load_images()
	train_dataset = dataset['train_dataset']
	validation_dataset = dataset['validation_dataset']

	print train_dataset.images.shape
	print validation_dataset.images.shape	
