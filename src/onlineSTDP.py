import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
np.random.seed(seed=0)

dt = 1e-3
T = 0.5
nt = round(T/dt)
tau_p = tau_m = 2e-2
A_p = 0.01; A_m = 1.05*A_p

spike_pre = np.zeros(nt); spike_pre[[50, 200, 225, 300, 425]] = 1
spike_post = np.zeros(nt); spike_post[[100, 150, 250, 350, 400]] = 1

x_pre_arr = np.zeros(nt); x_post_arr = np.zeros(nt)
w_arr = np.zeros(nt)

x_pre = x_post = 0
w = 0

for t in range(nt):
	x_pre = x_pre*(1-dt/tau_p) + spike_pre[t]
	x_post = x_post*(1-dt/tau_m) + spike_post[t]
	dw = A_p*x_pre*spike_post[t] - A_m*x_post*spike_pre[t]
	w += dw

	x_pre_arr[t] = x_pre
	x_post_arr[t] = x_post
	w_arr[t] = w

time = np.arange(nt)*dt*1e3
def hide_ticks():
	plt.gca().spines['right'].set_visible(False)
	plt.gca().spines['top'].set_visible(False)
	plt.gca().yaxis.set_ticks_position('left')
	plt.gca().xaxis.set_ticks_position('bottom')
plt.figure(figsize=(12, 12))
plt.subplot(5,1,1)
plt.plot(time, x_pre_arr, color="k")
plt.ylabel("$x_{pre}$"); hide_ticks(); plt.xticks([])
plt.subplot(5,1,2)
plt.plot(time, spike_pre, color="k")
plt.ylabel("pre- spikes"); hide_ticks(); plt.xticks([])
plt.subplot(5,1,3)
plt.plot(time, spike_post, color="k")
plt.ylabel("post- spikes"); hide_ticks(); plt.xticks([])
plt.subplot(5,1,4)
plt.plot(time, x_post_arr, color="k")
plt.ylabel("$x_{post}$"); hide_ticks(); plt.xticks([])
plt.subplot(5,1,5)
plt.plot(time, w_arr, color="k")
plt.xlabel("$t$ (ms)"); plt.ylabel("$w$"); hide_ticks()
plt.savefig('results/onlineSTDP.png')