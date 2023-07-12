from tqdm import tqdm

from transport_network_optimization.transport_problem import DualOracle, OptimParams

import numpy as np


class ACRCDOracleStacker:
    def __init__(self, oracle: DualOracle, graph, sources, targets):
        self.oracle = oracle
        self.graph = graph
        self.sources = sources
        self.targets = targets

        self.T_LEN = oracle.edges_num
        self.LA_LEN = oracle.zones_num
        self.MU_LEN = oracle.zones_num

        self.t_vector_size = self.T_LEN
        self.la_mu_vector_size = self.LA_LEN + self.MU_LEN

        self.t = oracle.t_bar.copy()
        self.la = np.zeros(oracle.zones_num)
        self.mu = np.zeros(oracle.zones_num)
        self.optim_params = OptimParams(self.t, self.la, self.mu)

    def __call__(self, t_block, la_mu_block, *args, **kwargs):
        assert (len(t_block)) == self.t_vector_size
        assert (len(la_mu_block)) == self.la_mu_vector_size

        self.optim_params.t = t_block
        self.optim_params.la = la_mu_block[:self.LA_LEN]
        self.optim_params.mu = la_mu_block[self.LA_LEN:]

        T, pred_maps = self.oracle.get_T_and_predmaps(self.graph, self.optim_params, self.sources, self.targets)
        self.d = self.oracle.get_d(self.optim_params, T)
        grad_t = self.oracle.get_flows_on_shortest(self.sources, self.targets, self.d, pred_maps)
        grad_la = self.oracle.grad_dF_dla(self.d)
        grad_mu = self.oracle.grad_dF_dmu(self.d)

        #         print(grad_t.shape, grad_la.shape, grad_mu.shape)
        la_mu_grad = np.hstack([grad_la, grad_mu])
        t_grad = np.hstack([grad_t])
        dual_value = self.oracle.calc_F(self.optim_params, T)

        self.flows = grad_t.copy()
        print('dual_value', dual_value)
        return t_grad, la_mu_grad, dual_value, self.flows

    def get_prime_value(self):
        return self.oracle.prime(self.flows, self.d)

    def get_init_vars_block(self):
        return np.hstack([self.oracle.t_bar.copy()]), np.hstack([np.zeros(self.LA_LEN), np.zeros(self.MU_LEN)])


# ACRCD
# y (paper) = q(code_)
def ACRCD_star(oracle_stacker: ACRCDOracleStacker, x1_0, x2_0, K, L1_init=5000, L2_init=5000):
    ADAPTIVE_DELTA = 1e-8

    history = []
    la_mu_grad_norms = []
    t_grad_norms = []

    x1_list = [x1_0]
    x2_list = [x2_0]

    z1 = y1 = x1_0
    z2 = y2 = x2_0
    print(1)
  

    L1 = L1_init
    L2 = L2_init
    beta = 1 / 2

    for i in tqdm(range(K)):
        tau = 2 / (i + 2)

        x1 = tau * z1 + (1 - tau) * y1
        x2 = tau * z2 + (1 - tau) * y2

        *_x, res_x, flows = oracle_stacker(x1, x2)  # moved out of the inner loop
        history.append(res_x)
        la_mu_grad_norms.append(np.linalg.norm(_x[0]))
        print('la grad ', np.linalg.norm(_x[0]))
        t_grad_norms.append(np.linalg.norm(_x[1]))
        print('mu grad', np.linalg.norm(_x[1]))

        n_ = L1 ** beta + L2 ** beta
        index_p = np.random.choice([0, 1], p=[L1 ** beta / n_,
                                              L2 ** beta / n_])
        Ls = [L1, L2]
        Ls[index_p] /= 2

        # ADAPTIVE

        inequal_is_true = False
        xs = [x1, x2]
        sampled_gradient_x = _x[index_p]
        for _ in tqdm(range(100)):
            if index_p == 0:
                y1 = xs[index_p] - 1 / Ls[index_p] * sampled_gradient_x
                y2 = x2
            else:
                y2 = xs[index_p] - 1 / Ls[index_p] * sampled_gradient_x
                y1 = x1

            *_y, res_y, flows = oracle_stacker(y1, y2)
            inequal_is_true = 1 / (2 * Ls[index_p]) * np.linalg.norm(
                sampled_gradient_x) ** 2 <= res_x - res_y + ADAPTIVE_DELTA
            if inequal_is_true:
                break
            Ls[index_p] *= 2

        L1, L2 = Ls
        n_ = L1 ** beta + L2 ** beta
        alpha = (i + 2) / (2 * n_ ** 2)

        if index_p == 0:
            z1 = z1 - (1 / L1) * alpha * n_ * sampled_gradient_x

        if index_p == 1:
            z2 = z2 - (1 / L2) * alpha * n_ * sampled_gradient_x

        x1_list.append(x1)
        x2_list.append(x2)

    return history, la_mu_grad_norms, t_grad_norms, x1_list, x2_list, [L1, L2]



