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
	train = read_train_file(10)

	images = train.images
	targets = train.targets

	indices = np.random.choice(len(images), len(images)/2, replace=False)

	images[indices] = np.reshape(np.reshape(images[indices], (-1, 96, 96, 1))[:, :, ::-1, :], (-1, 9216))

	targets[indices, ::2] = targets[indices, ::2] * -1 + 95
	
	flip_indices = [(0, 2), (1, 3), (4, 8), (5, 9), (6, 10), (7, 11), (12, 16), (13, 17), (14, 18), (15, 19), (22, 24), (23, 25)]

	for a, b in flip_indices:
		targets[indices, a], targets[indices, b] = targets[indices, b], targets[indices, a]

	plot_image(images[0], targets[0])
	plot_image(images[indices[0]], targets[indices[0]])
	plt.show()
