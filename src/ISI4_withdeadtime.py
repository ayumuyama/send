import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
np.random.seed(seed=0)

dt = 1e-3; T = 1; nt = round(T/dt)
n_neurons = 10
tref = 5e-3
fr = 30
spikes = np.zeros((nt, n_neurons))
tlast = np.zeros(n_neurons)
for i in range(nt):
	s = np.where(np.random.rand(n_neurons) < fr*dt, 1, 0)
	spikes[i] = ((dt*i) > (tlast + tref))*s
	tlast = tlast*(1-s) + dt*i*s

print("Num.of spikes:", np.sum(spikes))
print("Firing rate:", np.sum(spikes) / (n_neurons*T))
t = np.arange(nt)*dt
plt.figure(figsize=(10, 8))
for i in range(n_neurons):
	plt.plot(t, spikes[:, i]*(i+1), 'ko', markersize=2)
plt.xlabel('Time (s)'); plt.ylabel('Neuron index')
plt.xlim(0, T); plt.ylim(0.5, n_neurons+0.5)
plt.savefig('results/ISI4.png')
