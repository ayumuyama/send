import numpy as np
import matplotlib.pyplot as plt

dt = 1e-4; T = 0.05; nt = round(T/dt)
alpha = 1/5e-4; beta = 1/5e-3

r = 0; single_r = []
for t in range(nt):
	spike = 1 if t==0 else 0
	r += (alpha*spike*(1-r) - beta*r)*dt
	single_r.append(r)

time = np.arange(nt)*dt
plt.figure(figsize=(8, 6))
plt.plot(time, np.array(single_r), color="k")
plt.xlabel('Time (s)'); plt.ylabel('Post-synaptic current (pA)')
plt.savefig('results/KineticModel.png')
