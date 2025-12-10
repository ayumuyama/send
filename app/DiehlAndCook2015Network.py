import numpy as np
from app.Neurons import DiehlAndCook2015LIF

class DiehlAndCook2015Network:
	def __init__(self, n_in=784, n_neurons=100, wexc=2.25, winh=0.875, dt=1e-3, wmin=0.0, \
	                wmax=5e-2, lr=(1e-2, 1e-4), update_nt=100):
        self.dt = dt
        self.lr_p, self.lr_m = lr
        self.wmax = wmax
        self.wmin = wmin
        self.n_neurons = n_neurons
        self.n_in = n_in
        self.norm = 0.1
        self.update_nt = update_nt

        #Neuronsの設定
        self.exc_neurons = DiehlAndCook2015LIF(n_neurons, dt=dt, tref=2e-3, tc_m=1e-2, vrest=-60, \
                                                vreset=-45, vthr=-40, vpeak=20, e_exc=0, e_inh=-85)

        #Synapsesの設定
        self.input_synapse = SingleExponentialSynapse(n_in, dt=dt, td=1e-3)
        self.exc_synapse = SingleExponentialSynapse(n_neurons, dt=dt, td=1e-3)
        self.inh_synapse = SingleExponentialSynapse(n_neurons, dt=dt, td=2e-3)

        #重みの初期化
        self.input_synaptictrace = SingleExponentialSynapse(n_in, dt=dt, td=2e-2)
        self.exc_synaptictrace = SingleExponentialSynapse(n_neurons, dt=dt, td=2e-2)

        initW = 1e-3*np.random.rand(n_neurons, n_in)
        self.input_conn = FullConnection(n_in, n_neurons, initW=initW)
        self.exc2inh_W = wexc*np.eye(n_neurons)
        self.inh2exc_W = (winh/n_neurons)*(np.ones((n_neurons, n_neurons)) \
                                                - np.eye(n_neurons))

        self.delay_input = DelayConnection(N=n_neurons, delay=5e-3, dt=dt)
        self.delay_exc2inh = DelayConnection(N=n_neurons, delay=2e-3, dt=dt)
        self.g_inh = np.zeros(n_neurons)

        self.tcount = 0

        self.s_in_ = np.zeros((self.update_nt, n_in)) #複数次元を渡すとき、引数はタプル
        self.s_exc_ = np.zeros((n_neurons, self.update_nt))
        self.x_in_ = np.zeros((self.update_nt, n_in))
        self.x_exc_ = np.zeros((n_neurons, self.update_nt))

    #スパイクトレースのリセット
    def reset_trace(self):
        self.s_in_ = np.zeros((self.update_nt, self.n_in))
        self.s_exc_ = np.zeros((self.n_neurons, self.update_nt))
        self.x_in_ = np.zeros((self.update_nt, self.n_in))
        self.x_exc_ = np.zeros((self.n_neurons, self.update_nt))
        self.tcount = 0

    def initialize_states(self):
        self.exc_neurons.initialize_states()
        self.inh_neurons.initialize_states()
        self.delay_input.initialize_states()
        self.delay_exc2inh.initialize_states()
        self.input_synapse.initialize_states()
        self.exc_synapse.initialize_states()
        self.inh_synapse.initialize_states()

    def __call__(self, s_in, stdp=True):
        #入力層
        c_in = self.input_synapse(s_in)
        x_in = self.input_synaptictrace(s_in)
        g_in = self.input_conn(c_in)

        #興奮性ニューロン層
        s_exc = self.exc_neurons(self.delay_input(g_in), self.g_inh)
        c_exc = self.exc_synapse(s_exc)
        g_exc = np.dot(self.exc2inh_W, c_exc)
        x_exc = self.exc_synaptictrace(s_exc)

        #抑制性ニューロン層
        s_inh = self.inh_neurons(self.delay_exc2inh(g_exc), 0)
        c_inh = self.inh_synapse(s_inh)
        self.g_inh = np.dot(self.inh2exc_W, c_inh)

        if stdp:
            #スパイク列、スパイクトレースを記録
            self.s_in_[self.tcount] = s_in
            self.s_exc_[:, self.tcount] s_exc
            self.x_in_[self.tcount] = x_in
            self.x_exc_[:, self.tcount] = x_exc
            self.tcount += 1

            #Online STDP
            if self.tcount == self.update_nt:
                W = np.copy(self.input_conn.W)

                W_abs_sum = np.expand_dims(np.sum(np.abs(W), axis=1), 1)
                W_abs_sum[W_abs_sum == 0] = 1.0
                W *= self.norm / W_abs_sum
                #STDP則
                dW = self.lr_p*(self.wmax-W)*np.dot(self.s_exc_, self.x_in_)
                dW -= self.lr_m*W*np.dot(self.x_exc_, self.s_in_)
                clipped_dW = np.clip(dW / self.update_nt, -5e-2, 5e-2)
                self.input_conn.W = np.clip(W + clipped_dW, self.wmin, self.wmax)
                self.reset_trace()
        return s_exc
