import numpy as np
import matplotlib.pyplot as plt
from facial_keypoints_detection_input import *

file_path = 'dataset/'

def plot_image(image, target=None, title=None):
	plt.figure()

	plt.imshow(np.reshape(image, (96, 96)), cmap='gray')

	if target is not None:
		plt.plot(target[0::2], target[1::2], 'o')
		plt.title(title)

	return plt

def plot_loss_graph():
	train_validation_loss = load_train_validation_loss()

	train_loss = train_validation_loss['train_loss']
	validation_loss = train_validation_loss['validation_loss']

	plt.plot(train_loss, linewidth=3, label='Train')
	plt.plot(validation_loss, linewidth=3, label='Validation')

	plt.grid()
	plt.legend()
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.yscale('log')

	return plt

if __name__ == '__main__':
	batch_size = 10

	train = read_train_file(batch_size)

	images = train.images
	targets = train.targets

	horizontal_flipped_images = np.reshape(np.reshape(images, (-1, 96, 96, 1))[:, :, ::-1, :], (-1, 9216))

	images = np.append(images, horizontal_flipped_images, axis=0)

	targets = np.append(targets, targets, axis=0)
	targets[batch_size:, ::2] = targets[batch_size:, ::2] * -1 + 95
	
	flip_indices = [(0, 2), (1, 3), (4, 8), (5, 9), (6, 10), (7, 11), (12, 16), (13, 17), (14, 18), (15, 19), (22, 24), (23, 25)]

	for a, b in flip_indices:
		targets[batch_size:, [a, b]] = targets[batch_size:, [b, a]]

	plot_image(images[0], targets[0])
	plot_image(images[10], targets[10])
	plt.show()
