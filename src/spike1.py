import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
np.random.seed(seed=0)
from app.Hodgkin import HodgkinHuxleyModel

dt = 0.01; T = 400
nt = round(T/dt)
time = np.arange(0.0, T, dt)

I_inj = 10*(time>100) - 10*(time>200) + 35*(time>250) - 35*(time>350)
HH_neuron = HodgkinHuxleyModel(dt=dt, solver="Euler")
X_arr = np.zeros((nt, 4))

for i in tqdm(range(nt)):
	X = HH_neuron(I_inj[i])
	X_arr[i] = X

plt.figure(figsize=(5,5))
plt.subplot(3,1,1)
plt.plot(time,X_arr[:,0],color="k")
plt.ylabel('V(mV)'); plt.xlim(0,T)
plt.subplot(3,1,2)
plt.plot(time,I_inj, color="k")
plt.ylabel('$I_{inj}$($\\mu{A}/cm^2$)')
plt.xlim(0,T)
plt.subplot(3,1,3)
plt.plot(time,X_arr[:,1],'k',label='m')
plt.plot(time,X_arr[:,2],'gray', label='h')
plt.plot(time,X_arr[:,3],'k',linestyle="dashed",label='n')
plt.xlabel('t(ms)'); plt.ylabel('Gating Value');plt.legend(loc="upper left")
plt.savefig('results/spike1.png')

