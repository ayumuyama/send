import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
np.random.seed(seed=0)

dt = 0.5; T = 500
nt = round(T/dt)

C = 100
a = 0.03
b = -2
k = 0.7
d = 100
vrest = -60
vreset = -50
vthr = -40
vpeak = 35
t = np.arange(nt)*dt
I = 100*(t>50) - 100*(t>350)

v = vrest; v_ = v; u = 0
v_arr = np.zeros(nt)
u_arr = np.zeros(nt)

for i in tqdm(range(nt)):
	dv = (k*(v - vrest)*(v - vthr) - u + I[i]) / C
	v = v + dt*dv
	u = u + dt*(a*(b*(v_-vrest)-u))

	s = 1*(v>=vpeak)

	u = u + d*s
	v = v*(1-s) + vreset*s
	v_ = v

	v_arr[i] = v
	u_arr[i] = u

t = np.arange(nt)*dt
plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.plot(t, v_arr, color="k")
plt.ylabel('Membrane potential (mV)')

plt.subplot(2, 1, 2)
plt.plot(t, u_arr, color="k")
plt.xlabel('Time (ms)')
plt.ylabel('Recovery current (pA)')
plt.savefig('results/Iz1.png')
