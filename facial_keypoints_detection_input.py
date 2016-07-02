import csv
import numpy as np
from sklearn import cross_validation

file_path = 'dataset/'
train_file = 'training.csv'
test_file = 'test.csv'

class facial:
	pass

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

if __name__ == '__main__':
	dataset = load_images()
	train_dataset = dataset['train_dataset']
	validation_dataset = dataset['validation_dataset']

	print train_dataset.images.shape
	print validation_dataset.images.shape	
