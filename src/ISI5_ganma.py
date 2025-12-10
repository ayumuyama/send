import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
np.random.seed(seed=0)
import scipy.special as sps

dt = 1e-3; T = 1; nt = round(T/dt)
n_neurons = 10
fr = 30
k = 12
theta = 1/(k*(fr*dt))
isi = np.random.gamma(shape=k, scale=theta, \
			size=(round(nt*1.5/fr), n_neurons))
spike_time = np.cumsum(isi, axis=0)
spike_time[spike_time > nt - 1] = -1
spike_time = spike_time.astype(np.int32)
spikes = np.zeros((nt, n_neurons))
for i in range(n_neurons):
	spikes[spike_time[:, i], i] = 1
spikes[0] = 0
print("Num. of spikes:", np.sum(spikes))
print("Firing rate:", np.sum(spikes)/(n_neurons*T))

plt.figure(figsize=(10, 10))
t = np.arange(nt)*dt
plt.subplot(2,1,1)
count, bins, ignored = plt.hist(isi.flatten(), \
				50, density=True, \
				color="gray", alpha=0.5)
y = bins**(k-1)*(np.exp(-bins/theta) / (sps.gamma(k)*theta**k))
plt.plot(bins, y, linewidth=2, color="k")
plt.title('k='+str(k))
plt.xlabel('ISI (ms)')
plt.ylabel('Probability density')

plt.subplot(2,1,2)
for i in range(n_neurons):
	plt.plot(t, spikes[:, i]*(i+1), 'ko', markersize=2)
plt.xlabel('Time (s)'); plt.ylabel('Neuron index')
plt.xlim(0, T); plt.ylim(0.5, n_neurons+0.5)
plt.savefig('results/ISI5.png')
