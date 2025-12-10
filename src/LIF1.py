import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
np.random.seed(seed=0)

dt = 5e-5; T = 0.4
nt = round(T/dt)

tref = 2e-3
tc_m = 1e-2
vrest = -60
vreset = -65
vthr = -40
vpeak = 30

t = np.arange(nt)*dt*1e3
I = 25*(t>50) - 25*(t>350)

v = vreset
tlast = 0
v_arr = np.zeros(nt)

for i in tqdm(range(nt)):
	dv = (vrest - v + I[i]) / tc_m
	v = v + ((dt*i) > (tlast + tref))*dv*dt

	s = 1*(v>=vthr)
	tlast = tlast*(1-s) + dt*i*s
	v = v*(1-s) + vpeak*s
	v_arr[i] = v
	v = v*(1-s) + vreset*s

plt.figure(figsize=(5, 3))
plt.plot(t, v_arr)
plt.xlim(0, t.max())
plt.xlabel('Time (ms)'); plt.ylabel('Membrane potential (mV)')
plt.savefig('results/LIF1.png')
