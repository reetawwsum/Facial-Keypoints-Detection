import csv
import numpy as np
import matplotlib.pyplot as plt

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
			if csv_reader.line_num == max_images:
				break

			targets.append(row[:-1])
			images.append(row[-1].split())

		images = np.array(images).astype(int)

	return images, targets

def read_test_file(max_images=None):
	with open(file_path + test_file, 'rb') as f:
		csv_reader = csv.reader(f, delimiter=',')

		csv_reader.next()
		images = []

		for row in csv_reader:
			if csv_reader.line_num == max_images:
				break

			images.append(row[-1].split())
		
		images = np.array(images).astype(int)

	return images


def plot_image(image, target=None):
	plt.figure()

	plt.imshow(np.reshape(image, (96, 96)), cmap='gray')

	if target is not None:
		plt.plot(target[0::2], target[1::2], 'o')

	return plt

if __name__ == '__main__':
	train_images, targets = read_train_file(10)
	test_images = read_test_file(10)

	random_train_index = np.random.randint(train_images.shape[0])
	random_test_index = np.random.randint(test_images.shape[0])

	plot_image(train_images[random_train_index], targets[random_train_index])
	plot_image(test_images[random_test_index])

	plt.show()
