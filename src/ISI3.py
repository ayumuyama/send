import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
np.random.seed(seed=0)

dt = 1e-3; T = 1; nt = round(T/dt)
n_neuron = 10
t = np.arange(nt)*dt

fr = np.expand_dims(30*np.sin(10*t)**2, 1)

spikes = np.where(np.random.rand(nt, n_neuron) < fr*dt, 1, 0)

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(t, fr[:, 0], color="k")
plt.ylabel('Firing rate (Hz)')
plt.xlim(0, T)

plt.subplot(2, 1, 2)
for i in range(n_neuron):
	plt.plot(t, spikes[:, i]*(i+1), 'ko', markersize=2, \
			rasterized=True)
plt.xlabel('Time (s)'); plt.ylabel('Neuron index')
plt.xlim(0, T); plt.ylim(0.5, n_neuron+0.5)
plt.savefig('results/ISI3.png')

