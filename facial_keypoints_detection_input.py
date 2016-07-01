import csv
import numpy as np

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

if __name__ == '__main__':
	images, targets = read_train_file()

	print images.shape
