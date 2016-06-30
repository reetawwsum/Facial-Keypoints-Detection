import csv
import numpy as np
import matplotlib.pyplot as plt

file_path = 'dataset/'
train_file = 'training.csv'

with open(file_path + train_file, 'rb') as f:
	csv_reader = csv.reader(f, delimiter=',')

	csv_reader.next()
	targets = []
	images = []

	for row in csv_reader:
		targets.append(row[:-1])
		images.append(row[-1].split())

images = np.array(images).astype(int)

image = images[0]
target = targets[0]

plt.imshow(np.reshape(images[0], (96, 96)), cmap='gray')
plt.plot(target[0::2], target[1::2], 'o')
plt.show()
