import numpy as np

class CurrentBasedLIF:
	def __init__(self, N, dt=1e-4, tref=5e-3, tc_m=1e-2, \
			vrest=-60, vreset=-60, vthr=-50, vpeak=20):
		self.N = N
		self.dt = dt
		self.tref = tref
		self.tc_m = tc_m
		self.vrest = vrest
		self.vreset = vreset
		self.vthr = vthr
		self.vpeak = vpeak

		self.v = self.vreset*np.ones(N)
		self.v_ = None
		self.tlast = 0
		self.tcount = 0

	def initialize_states(self, random_state=False):
		if random_state:
			self.v = self.vreset + \
				np.random.rand(self.N)*(self.vthr-self.vreset)

		else:
			self.v = self.vreset*np.ones(self.N)

		self.tlast = 0
		self.tcount = 0

	def __call__(self, I):
		dv = (self.vrest - self.v + I) / self.tc_m
		v = self.v + (self.dt*self.tcount>(self.tlast+self.tref))*dv*self.dt

		s = 1*(v>=self.vthr)
		self.tlast = self.tlast*(1-s) + self.dt*self.tcount*s
		v = v*(1-s) + self.vpeak*s
		self.v_ = v
		self.v = v*(1-s) + self.vreset*s
		self.tcount += 1

		return s

class IzhikevichNeuron:
	def __init__(self, N, dt=0.5, C=250, a=0.01, b=-2, \
			k=2.5, d=200, vrest=-60, vreset=-65, vthr=-20, vpeak=30):
		self.N = N
		self.dt = dt
		self.C = C
		self.a = a
		self.b = b
		self.d = d
		self.k = k
		self.vrest = vrest
		self.vreset = vreset
		self.vthr = vthr
		self.vpeak = vpeak

		self.u = np.zeros(N)
		self.v = self.vrest*np.ones(N)
		self.v_ = self.v

	def initialize_states(self, random_state=False):
		if random_state:
			self.v = self.vreset + np.random.rand(self.N)\
					*(self.vthr-self.vreset)
		else:
			self.v = self.vrest*np.ones(self.N)
		self.u = np.zeros(self.N)

	def __call__(self, I):
		dv = (self.k*(self.v-self.vrest)*(self.v-self.vthr)-self.u+I)/self.C
		v = self.v + self.dt*dv
		u = self.u + self.dt*(self.a*(self.b*(self.v_-self.vrest)-self.u))

		s = 1*(v>=self.vpeak)
		self.u = u + self.d*s
		self.v = v*(1-s) + self.vreset*s
		self.v_ = self.v
		return s

class DiehlAndCook2015LIF:
	def __init__(self, N, dt=1e-3, tref=5e-3, tc_m=1e-1, vrest=-65, vreset=-65, \
			init_vthr=-52, vpeak=20, theta_plus=0.05, theta_max=35, tc_theta=1e4, \
			e_exc=0, e_inh=-100):
		self.N = N
		self.dt = dt
		self.tref = tref
		self.tc_m = tc_m
		self.vreset = vreset
		self.vrest = vrest
		self.init_vthr = init_vthr
		self.theta = np.zeros(N)
		self.theta_plus = theta_plus
		self.theta_max = theta_max
		self.tc_theta = tc_theta
		self.vpeak = vpeak

		self.e_exc = e_exc
		self.e_inh = e_inh

		self.v = self.vreset*np.ones(N)
		self.vthr = self.init_vthr
		self.v_ = None
		self.tlast = 0
		self.tcount = 0

	def initialize_states(self, random_state=False):
		if random_state:
			self.v = self.vreset + np.random.rand(self.N)*(self.vthr-self.vreset)

		else:
			self.v = self.vreset*np.ones(self.N)
		self.vthr = self.init_vthr
		self.theta = np.zeros(self.N)
		self.tlast = 0
		self.tcount = 0


	def __call__(self, g_exc, g_inh):
		I_synExc = g_exc*(self.e_exc - self.v)
		I_synInh = g_inh*(self.e_inh - self.v)
		dv = (self.vrest - self.v + I_synExc + I_synInh) / self.tc_m
		v = self.v+((self.dt*self.tcount)>(self.tlast+self.tref))*dv*self.dt

		s = 1*(v>=self.vthr)

		theta = (1-self.dt/self.tc_theta)*self.theta + self.theta_plus*s #スパイクが発生すると、閾値が上昇し、時間とともに減衰する。
		self.theta = np.clip(theta, 0, self.theta_max) #配列thetaの最小値を0に、最大値をtheta_maxにする
		self.vthr = self.theta + self.init_vthr

		self.tlast = self.tlast*(1-s) + self.dt*self.tcount*s
		v = v*(1-s) + self.vpeak*s
		self.v_ = v
		self.v = v*(1-s) + self.vreset*s
		self.tcount += 1

		return s
