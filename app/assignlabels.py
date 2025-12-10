import numpy as np

def assign_labels(spikes, labels, n_labels, rates=None, alpha=1.0):
	n_neurons = spikes.shape[1]

	if rates is None:
		rates = np.zeros((n_neurons, n_labels)).astype(np.float32)

	for i in range(n_labels):
		n_labeled = np.sum(labels == i).astype(np.int16)

		if n_labeled > 0:
			indices = np.where(labels == i)[0]

			rates[:, i] = alpha*rates[:, i] + \
					(np.sum(spikes[indices], axis=0)/n_labeled)

	sum_rate = np.sum(rates, axis=1)
	sum_rate[sum_rate==0] = 1
	proportions = rates / np.expand_dims(sum_rate, 1)
	proportions[proportions != proportions] = 0

	assignments = np.argmax(proportions, axis=1).astype(np.unit8)

	return assignments, proportions, rates

def prediction(spikes, assignments, n_labels):
	n_samples = spikes.shape[0]

	rates = np.zeros((n_samples,n_labels)).astype(np.float32)

	for i in range(n_labels):

		n_assigns = np.sum(assignments==i).astype(np.uint8)

		if n_assigns>0:

			indices = np.where(assignments == i)[0]
			rates[:, i] = np.sum(spikes[:, indices], axis=1) / n_assigns

	return np.argmax(rates, axis=1).astype(np.uint8)