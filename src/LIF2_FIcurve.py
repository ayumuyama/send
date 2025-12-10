import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
np.random.seed(seed=0)

dt = 5e-5; T = 1; nt = round(T/dt)
tref = 5e-3; tc_m = 1e-2; vrest = 0; vreset = 0; vthr = 1

I_max = 3
N = 100
I = np.linspace(0, I_max, N)
spikes  = np.zeros((N, nt))

for i in tqdm(range(N)):
	v = vreset; tlast = 0
	for t in range(nt):
		dv = (vrest-v + I[i]) / tc_m
		update = 1 if ((dt*t) > (tlast + tref)) else 0
		v = v + update*dv*dt
		s = 1 if (v>=vthr) else 0
		tlast = tlast*(1-s) + dt*t*s
		spikes[i, t] = s
		v = v*(1-s) + vreset*s

rate = np.sum(spikes, axis=1) / T
plt.figure(figsize=(4,3))
plt.plot(I, rate)
plt.xlabel('Input current (nA)'); plt.ylabel('Firing rate(Hz)')
plt.savefig('results/LIF_FIcurve.png')
