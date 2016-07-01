import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

file_path = 'dataset/'
train_file = 'training.csv'
test_file = 'test.csv'

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

	return images, targets

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

	return images

def preprocessing_dataset(raw_images, raw_targets):
	images = MinMaxScaler(feature_range=(0, 1)).fit_transform(raw_images)
	targets = MinMaxScaler(feature_range=(-1, 1)).fit_transform(raw_targets)

	return images, targets

def plot_image(image, target=None):
	plt.figure()

	plt.imshow(np.reshape(image, (96, 96)), cmap='gray')

	if target is not None:
		plt.plot(target[0::2], target[1::2], 'o')

	return plt

if __name__ == '__main__':
	raw_images, raw_targets = read_train_file()
	images, targets = preprocessing_dataset(raw_images, raw_targets)

	print images.shape
