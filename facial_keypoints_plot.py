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
	plot_loss_graph()
	plt.show()	
