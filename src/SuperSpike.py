import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
np.random.seed(seed=0)

from app.Neurons import CurrentBasedLIF
from app.Synapses import DoubleExponentialSynapse
from app.Connections import FullConnection, DelayConnection
from app.ErrorSignal import ErrorSignal, EligibilityTrace

dt = 1e-4; T = 0.5; nt = round(T/dt)

t_weight_update = 0.5
nt_b = round(t_weight_update/dt)

num_iter = 200

N_in = 50
N_mid = 4
N_out = 1

fr_in = 10
x = np.where(np.random.rand(nt, N_in) < fr_in * dt, 1, 0) #発火率に基づいたスパイクの生成
y = np.zeros((nt, N_out))
y[int(nt/10)::int(nt/5), :] = 1 #1行上で作成した0行列に開始(nt/10)(delayに合わせた猶予？)で(nt/5)ごとに全列に対して1を代入


neurons_1 = CurrentBasedLIF(N_mid, dt=dt)
neurons_2 = CurrentBasedLIF(N_out, dt=dt)
delay_conn1 = DelayConnection(N_in, delay=8e-4)
delay_conn2 = DelayConnection(N_mid, delay=8e-4)
synapses_1 = DoubleExponentialSynapse(N_in, dt=dt, td=1e-2, tr=5e-3)
synapses_2 = DoubleExponentialSynapse(N_mid, dt=dt, td=1e-2, tr=5e-3)
es = ErrorSignal(N_out)
et1 = EligibilityTrace(N_in, N_mid)
et2 = EligibilityTrace(N_mid, N_out)

connect_1 = FullConnection(N_in, N_mid, initW=0.1*np.random.rand(N_mid, N_in))
connect_2 = FullConnection(N_mid, N_out, initW=0.1*np.random.rand(N_out, N_mid))

r0 = 1e-3
gamma = 0.7

current_arr = np.zeros((N_mid, nt))
voltage_arr = np.zeros((N_out, nt))
error_arr = np.zeros((N_out, nt))
lambda_arr = np.zeros((N_out, N_mid, nt)) #N_out個のN_mid行nt列の行列
dw_arr = np.zeros((N_out, N_mid, nt))
cost_arr = np.zeros(num_iter)

for i in tqdm(range(num_iter)):
	if i%15 == 0:
		r0 /= 2 #重みの減衰（純粋に試行回数が増えるにつれ減らす形？）

	neurons_1.initialize_states()
	neurons_2.initialize_states()
	synapses_1.initialize_states()
	synapses_2.initialize_states()
	es.initialize_states()
	et1.initialize_states()
	et2.initialize_states()

	m1 = np.zeros((N_mid, N_in))
	m2 = np.zeros((N_out, N_mid))
	v1 = np.zeros((N_mid, N_in))
	v2 = np.zeros((N_out, N_mid))
	cost = 0
	count = 0

	#one iter
	for t in range(nt):
		#Feed foward
		c1 = synapses_1(delay_conn1(x[t])) #ポアソン過程のスパイクに遅延を与え、シナプスに伝達
		h1 = connect_1(c1) #上層シナプスへの入力生成
		s1 = neurons_1(h1)

		c2 = synapses_2(delay_conn2(s1))
		h2 = connect_2(c2)
		s2 = neurons_2(h2)

		#Backward
		e2 = np.expand_dims(es(s2, y[t]), axis=1) / N_out
		e1 = connect_2.backward(e2) / N_mid

		cost += np.sum(e2**2) #意図わからん

		lambda2 = et2(c2, neurons_2.v_) #意図わからん2多分電流と電圧与えて、どれだけ重みの更新に役立ったか書いているはず
		lambda1 = et1(c1, neurons_1.v_)

		g2 = e2 * lambda2
		g1 = e1 * lambda1

		v1 = np.maximum(gamma*v1, g1**2)
		v2 = np.maximum(gamma*v2, g2**2)

		m1 += g1
		m2 += g2

		count += 1
		if count == nt_b:
			lr1 = r0/np.sqrt(v1+1e-8)
			lr2 = r0/np.sqrt(v2+1e-8)
			dW1 = np.clip(lr1*m1*dt, -1e-3, 1e-3)
			dW2 = np.clip(lr2*m2*dt, -1e-3, 1e-3)
			connect_1.W = np.clip(connect_1.W+dW1, -0.1, 0.1)
			connect_2.W = np.clip(connect_2.W+dW2, -0.1, 0.1)

			m1 = np.zeros((N_mid, N_in))
			m2 = np.zeros((N_out, N_mid))
			v1 = np.zeros((N_mid, N_in))
			v2 = np.zeros((N_out, N_mid))
			count = 0

		if i == num_iter-1:
			current_arr[:,t] = c2
			voltage_arr[:,t] = neurons_2.v_
			error_arr[:,t] = e2
			lambda_arr[:, :, t] = lambda2

	cost_arr[i] = cost
	print("\n cost:", cost)


t = np.arange(nt)*dt*1e3
plt.figure(figsize=(8, 10))
plt.subplot(6, 1, 1)
plt.plot(t,voltage_arr[0])
plt.ylabel('Membrane\n potential(mV)')
plt.subplot(6, 1, 2)
plt.plot(t, et1.surrogate_derivative_fastsigmoid(u=voltage_arr[0]))
plt.ylabel('Surrogate derivative')
plt.subplot(6, 1, 3)
plt.plot(t,error_arr[0])
plt.ylabel('Error')
plt.subplot(6, 1, 4)
plt.plot(t,lambda_arr[0,0], color="k")
plt.ylabel('$\lambda$')
plt.subplot(6, 1, 5)
plt.plot(t, current_arr[0], color="k")
plt.ylabel('Input current (pA)')
plt.subplot(6, 1, 6)
for i in range(N_in):
	plt.plot(t, x[:,i]*(i+1), 'ko', markersize=2)
plt.xlabel('Time(ms)'); plt.ylabel('Neuron index')
plt.xlim(0, t.max()); plt.ylim(0.5, N_in+0.5)
plt.savefig('/root/results/SuperSpike1.png')
plt.figure(figsize=(8,6))
plt.plot(cost_arr, color="k")
plt.xlabel('Iter'); plt.ylabel('Cost')
plt.savefig('results/SuperSpike2.png')
