import sys

from tqdm import tqdm

from transport_problem import DualOracle, OptimParams

import numpy as np

from dataclasses import dataclass


@dataclass
class Log:
    history = []
    la_mu_grad_norms = []
    la_mu_calls = 0
    t_calls = 0
    t_grad_norms = []


log = Log()


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

    def t_step(self, t_block):
        assert (len(t_block)) == self.t_vector_size
        t_block = np.maximum(t_block, self.oracle.t_bar)
        self.optim_params.t = t_block
        T, pred_maps = self.oracle.get_T_and_predmaps(self.optim_params, self.sources, self.targets)

        # FIXME: dont call
        self.d = self.oracle.get_d(self.optim_params, T)

        flows_on_shortest = self.oracle.get_flows_on_shortest(self.sources, self.targets, self.d, pred_maps)
        grad_t = self.oracle.grad_dF_dt(self.optim_params, flows_on_shortest)

        t_grad = np.hstack([grad_t])
        log.t_grad_norms.append(np.linalg.norm(t_grad))
        dual_value = self.oracle.calc_F(self.optim_params, T)
        self.flows = flows_on_shortest.copy()
        log.t_calls += 1
        print(f"{dual_value=}")
        if np.isnan(dual_value):
            print(self.optim_params)
            print(T.shape)
            dual_value = self.oracle.calc_F(self.optim_params, T)
            sys.exit()
        return dual_value, t_grad, self.flows

    def la_mu_step(self, la_mu_block):
        assert (len(la_mu_block)) == self.la_mu_vector_size
        self.optim_params.la = la_mu_block[:self.LA_LEN]
        self.optim_params.mu = la_mu_block[self.LA_LEN:]

        # FIXME: dont call
        T, pred_maps = self.oracle.get_T_and_predmaps(self.optim_params, self.sources, self.targets)

        self.d = self.oracle.get_d(self.optim_params, T)
        grad_la = self.oracle.grad_dF_dla(self.d)
        grad_mu = self.oracle.grad_dF_dmu(self.d)
        la_mu_grad = np.hstack([grad_la, grad_mu])
        debug = np.linalg.norm(la_mu_grad)
        try:
            debug2 = log.la_mu_grad_norms[-1]
            if abs(log.la_mu_grad_norms[-1] - np.linalg.norm(la_mu_grad)) > 0.5:
                ...
        except IndexError:
            pass
        log.la_mu_grad_norms.append(np.linalg.norm(la_mu_grad))
        dual_value = self.oracle.calc_F(self.optim_params, T)
        # dual_value = self.oracle.calc_F_via_d(self.optim_params, self.d,T)
        log.la_mu_calls += 1
        if np.isnan(dual_value):
            print(self.optim_params)
            print(T.shape)
            sys.exit()
        print(f"{dual_value=}")
        return dual_value, la_mu_grad, self.d

    def __call__(self, t_block, la_mu_block, *args, **kwargs):
        # For backward compatibility
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
        print(la_mu_grad)
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
    ADAPTIVE_DELTA = 1e-3

    flows_averaged = np.zeros(oracle_stacker.oracle.edges_num)
    corrs_averaged = np.zeros(oracle_stacker.oracle.zones_num)
    steps_sum = [0, 0]

    x1_list = [x1_0]
    x2_list = [x2_0]

    z1 = y1 = x1_0
    z2 = y2 = x2_0

    z1_ = 0
    z2_ = 0

    L1 = L1_init
    L2 = L2_init
    beta = 1 / 2
    
    res_x, sampled_gradient_x, flows = oracle_stacker.la_mu_step(x2_0)
    print("######### ", np.linalg.norm(sampled_gradient_x))

    for i in tqdm(range(K)):
        tau = 2 / (i + 2)
        x1 = tau * z1 + (1 - tau) * y1
        x2 = tau * z2 + (1 - tau) * y2

        n_ = L1 ** beta + L2 ** beta
        print(f"{L1=}")
        print(f"{L2=}")
        index_p = np.random.choice([0, 1], p=[L1 ** beta / n_,
                                              L2 ** beta / n_])

        if index_p == 0:
            res_x, sampled_gradient_x, flows = oracle_stacker.t_step(x1)  # moved out of the inner loop
        elif index_p == 1:
            res_x, sampled_gradient_x, d = oracle_stacker.la_mu_step(x2)  # moved out of the inner loop

        Ls = [L1, L2]
        Ls[index_p] /= 2

        # ADAPTIVE
        xs = [x1, x2]
        # sampled_gradient_x = _x[0]
        for _ in range(100):
            if index_p == 0:
                y1 = np.maximum(xs[index_p] - 1 / Ls[index_p] * sampled_gradient_x, oracle_stacker.oracle.t_bar)
                # y1 = xs[index_p] - 1 / Ls[index_p] * sampled_gradient_x
                y2 = x2
            else:
                y2 = xs[index_p] - 1 / Ls[index_p] * sampled_gradient_x
                y1 = x1

            if index_p == 0:
                res_y, _y, flows = oracle_stacker.t_step(y1)
                if np.isnan(res_y):
                    sys.exit()
                print(f"{res_y=}")
            else:
                res_y, _y, d = oracle_stacker.la_mu_step(y2)
                print(f"{res_y=}")

            inequal_is_true = 1 / (2 * Ls[index_p]) * np.linalg.norm(
                sampled_gradient_x) ** 2 <= res_x - res_y + ADAPTIVE_DELTA
            print("1 ", 1 / (2 * Ls[index_p]) * np.linalg.norm(
                sampled_gradient_x) ** 2)

            print("2 ", res_x - res_y + ADAPTIVE_DELTA)
            print(f"{inequal_is_true=}")
            if inequal_is_true:
                break
            Ls[index_p] *= 2

        if index_p == 0:
            flows_averaged = (steps_sum[index_p] * flows_averaged + (1 / Ls[index_p]) * flows) / (
                        steps_sum[index_p] + 1 / Ls[index_p])
        else:
            corrs_averaged = (steps_sum[index_p] * corrs_averaged + (1 / Ls[index_p]) * d) / (
                        steps_sum[index_p] + 1 / Ls[index_p])

        steps_sum[index_p] += (1 / Ls[index_p])
        L1, L2 = Ls

        inequal_is_true = 1 / (2 * Ls[index_p]) * np.linalg.norm(
            sampled_gradient_x) ** 2 <= z1_ - z2_ + ADAPTIVE_DELTA
        
        if inequal_is_true:
            n_ = L1 ** beta + L2 ** beta
            alpha = (i + 2) / (2 * n_ ** 2)

            if index_p == 0:
                z1 = np.maximum(z1 - (1 / L1) * alpha * n_ * sampled_gradient_x, oracle_stacker.oracle.t_bar)
                z1_ = z1

            if index_p == 1:
                z2 = z2 - (1 / L2) * alpha * n_ * sampled_gradient_x
                z2_ = z2
        else:
            Ls[index_p] *= 2



        # z1, z2 = y1, y2

        x1_list.append(x1)
        x2_list.append(x2)
        debug = abs(oracle_stacker.oracle.prime(flows_averaged, corrs_averaged) + res_y)
        log.history.append(abs(oracle_stacker.oracle.prime(flows_averaged, corrs_averaged) + res_y))
        print(f"{log.t_calls=}")
        print(f"{log.la_mu_calls=}")
        print(f"{oracle_stacker.oracle.prime(flows_averaged, corrs_averaged)=}")

    return log.t_calls, log.la_mu_calls, log.history, log.la_mu_grad_norms, log.t_grad_norms, x1_list, x2_list, [L1, L2]
