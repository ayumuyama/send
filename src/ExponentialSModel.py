import numpy as np
import matplotlib.pyplot as plt

dt = 5e-5
td = 2e-2
tr = 2e-3
T = 0.1
nt = round(T/dt)

#単一関数型シナプス
r = 0
single_r = []
for t in range(nt):
	spike = 1 if t == 0 else 0
	single_r.append(r)
	r = r*(1-dt/td) + spike/td

#二重指数関数型シナプス
r = 0; hr = 0
double_r = []
for t in range(nt):
	spike = 1 if t == 0 else 0
	double_r.append(r)
	r = r*(1-dt/tr) + hr*dt
	hr = hr*(1-dt/td) + spike/(tr*td)

time = np.arange(nt)*dt
plt.figure(figsize=(8, 6))
plt.plot(time, np.array(single_r), label="single exponential")
plt.plot(time, np.array(double_r), label="double exponential")
plt.xlabel('Time (s)'); plt.ylabel('Post-synaptic current (pA)')
plt.legend()
plt.savefig('results/ExpoSModel.png')
