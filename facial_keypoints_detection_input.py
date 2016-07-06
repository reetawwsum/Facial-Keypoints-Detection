import csv
import numpy as np
from datetime import datetime
from six.moves import cPickle as pickle
from sklearn import cross_validation

file_path = 'dataset/'
train_file = 'training.csv'
test_file = 'test.csv'
log_file = 'log.csv'

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
