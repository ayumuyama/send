import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
np.random.seed(seed=0)

dt = 1e-3; T = 1; nt = round(T/dt)
n_neurons = 10
fr = 30
isi = np.random.exponential(1/(fr*dt), \
				size = (round(nt*1.5/fr), n_neurons))
#幅の余裕をもたせるための1.5倍
#sizeの第一引数をkとすると、isi[k, 1]はk回目とk+1回目の発火間隔

spike_time = np.cumsum(isi, axis=0)
#ニューロンごとに累積
spike_time[spike_time > nt - 1] = 0
spike_time = spike_time.astype(np.int32)
spikes = np.zeros((nt, n_neurons))
for i in range(n_neurons):
	spikes[spike_time[:, i], i] = 1
#:のみは全選択
#spike_timeは今ニューロンごとの指数分布の乱数値を累積したもの
#↑ニューロンiが発火した時刻？累積されてるから間隔が2sとかだったら1s飛ばされて1が入るはず
spikes[0] = 0
print("Num. of spikes:", np.sum(spikes))
print("Firing rate:", np.sum(spikes)/(n_neurons*T))

t = np.arange(nt)*dt
plt.figure(figsize=(10, 8))
for i in range(n_neurons):
	plt.plot(t, spikes[:, i]*(i+1), 'ko', markersize=2)
plt.xlabel('Time (s)'); plt.ylabel('Neuron index')
plt.xlim(0, T); plt.ylim(0.5, n_neurons+0.5)
plt.savefig('../results/ISI1.png')
