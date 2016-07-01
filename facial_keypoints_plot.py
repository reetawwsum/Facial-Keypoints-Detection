import numpy as np
import matplotlib.pyplot as plt
from facial_keypoints_detection_input import *

def plot_image(image, target=None):
	plt.figure()

	plt.imshow(np.reshape(image, (96, 96)), cmap='gray')

	if target is not None:
		plt.plot(target[0::2], target[1::2], 'o')

	return plt

if __name__ == '__main__':
	images, targets = read_train_file(10)

	random_index = np.random.randint(len(images))

	plot_image(images[random_index], targets[random_index])
	plt.show()
