import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
np.random.seed(seed=0)

tc_m = 1e-2
R = 1
vthr = 1
tref = 5e-3
I_max = 3
I = np.arange(0, I_max, 0.01)

rate = 1 / (tref + tc_m*np.log(R*I / (R*I - vthr)))
rate[np.isnan(rate)] = 0

plt.figure(figsize=(8, 6))
plt.plot(I, rate, color="k")
plt.xlabel('Input current (nA)'); plt.ylabel('Firing rate (Hz)')
plt.xlim(0, I_max)
plt.savefig('results/LIF3rate.png')
