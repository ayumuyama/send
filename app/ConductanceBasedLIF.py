import numpy as np

class ConductanceBasedLIF:
	def __init__(self, N, dt=1e-4, tref=5e-3, tc_m=1e-2, vrest=-60, vreset=-60, vthr=-50, vpeak=20, e_exc=0, e_inh=-100):
		self.N = N
		self.dt = dt
		self.tref = tref
		self.tc_m = tc_m
		self.vrest = vrest
		self.vreset = vreset
		self.vthr = vthr
		self.vpeak = vpeak

		self.e_exc = e_exc
		self.e_inh = e_inh

		self.v = self.vreset*np.ones(N)
		self.v_ = None
		self.tlast = 0
		self.tcount = 0

	def initialize_states(self, random_state=False):
		if random_state:
			self.v = self.vreset + np.random.rand(self.N)*(self.vthr-self.vreset)
		else:
			self.v = self.vreset*np.ones(self.N)
		self.tlast = 0
		self.tcount = 0

	def __call__(self, g_exc, g_inh):
		I_synExc = g_exc*(self.e_exc - self.v)
		I_synInh = g_inh*(self.e_inh - self.v)
		dv = (self.vrest -self.v + I_synExc + I_synInh) / self.tc_m
		v = self.v+((self.dt*self.tcount)>(self.tlast+self.tref))*dv*self.dt

		s = 1*(v>=self.vthr)

		self.tlast = self.tlast*(1-s) + self.dt*self.tcount*s
		v = v*(1-s) + self.vpeak*s
		self.v_ = v
		self.v = v*(1-s) + self.vreset*s
		self.tcount += 1

		return s
