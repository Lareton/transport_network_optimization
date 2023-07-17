import sys
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass, field

from transport_problem import DualOracle, OptimParams
from oracle_utils import AlgoResults


@dataclass
class Log:
    history: list = field(default_factory=list)
    la_mu_grad_norms: list = field(default_factory=list)
    la_mu_calls = 0
    t_calls = 0
    t_grad_norms: list = field(default_factory=list)


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
        dual_value = self.oracle.calc_F(self.optim_params, T)
        self.flows = flows_on_shortest.copy()

        if np.isnan(dual_value):
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

        dual_value = self.oracle.calc_F(self.optim_params, T)
        if np.isnan(dual_value):
            sys.exit()
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

        la_mu_grad = np.hstack([grad_la, grad_mu])
        t_grad = np.hstack([grad_t])
        dual_value = self.oracle.calc_F(self.optim_params, T)

        self.flows = grad_t.copy()
        return t_grad, la_mu_grad, dual_value, self.flows

    def get_prime_value(self):
        return self.oracle.prime(self.flows, self.d)

    def get_init_vars_block(self):
        return np.hstack([self.oracle.t_bar.copy()]), np.hstack([np.zeros(self.LA_LEN), np.zeros(self.MU_LEN)])


# ACRCD
# y (paper) = q(code_)
def ACRCD_star(oracle_stacker: ACRCDOracleStacker, x1_0, x2_0, K, L1_init=1e-1, L2_init=1e-1):
    global log
    results_t = AlgoResults()
    results_la_mu = AlgoResults()

    flows_averaged = np.zeros(oracle_stacker.oracle.edges_num)
    corrs_averaged = np.zeros(oracle_stacker.oracle.zones_num)
    steps_sum = [0, 0]

    x1_list = [x1_0]
    x2_list = [x2_0]

    z1 = y1 = x1_0
    z2 = y2 = x2_0

    L1 = L1_init
    L2 = L2_init
    beta = 1 / 2

    res_x, sampled_gradient_x, flows = oracle_stacker.la_mu_step(x2_0)

    for i in tqdm(range(K)):
        tau = 2 / (i + 2)
        x1 = tau * z1 + (1 - tau) * y1
        x2 = tau * z2 + (1 - tau) * y2

        n_ = L1 ** beta + L2 ** beta
        index_p = np.random.choice([0, 1], p=[L1 ** beta / n_,
                                              L2 ** beta / n_])

        if index_p == 0:
            res_x, sampled_gradient_x, flows = oracle_stacker.t_step(x1)  # moved out of the inner loop

            _, gradient_la_mu, _ = oracle_stacker.la_mu_step(x2)  # FOR TESTING ONLY!!!

            if i > 5:
                results_t.count_oracle_calls += 1
                results_t.history_dual_gap.append(
                    abs(oracle_stacker.oracle.prime(flows_averaged, corrs_averaged) + res_x))
                results_t.history_count_calls.append(results_t.count_oracle_calls)
                results_t.history_la_mu_grad_norm.append(np.linalg.norm(gradient_la_mu))


        elif index_p == 1:
            res_x, sampled_gradient_x, d = oracle_stacker.la_mu_step(x2)  # moved out of the inner loop

            if i > 5:
                results_la_mu.count_oracle_calls += 1
                results_la_mu.history_la_mu_grad_norm.append(np.linalg.norm(sampled_gradient_x))
                results_la_mu.history_dual_gap.append(
                    abs(oracle_stacker.oracle.prime(flows_averaged, corrs_averaged) + res_x))
                results_la_mu.history_count_calls.append(results_la_mu.count_oracle_calls)

        Ls = [L1, L2]

        if index_p == 0:
            flows_averaged = (steps_sum[index_p] * flows_averaged + (1 / Ls[index_p]) * flows) / (
                    steps_sum[index_p] + 1 / Ls[index_p])
        else:
            corrs_averaged = (steps_sum[index_p] * corrs_averaged + (1 / Ls[index_p]) * d) / (
                    steps_sum[index_p] + 1 / Ls[index_p])

        steps_sum[index_p] += (1 / Ls[index_p])

        L1, L2 = Ls

        n_ = L1 ** beta + L2 ** beta
        alpha = (i + 2) / (2 * n_ ** 2)

        if index_p == 0:
            z1 = np.maximum(z1 - (1 / L1) * alpha * n_ * sampled_gradient_x, oracle_stacker.oracle.t_bar)

        if index_p == 1:
            z2 = z2 - (1 / L2) * alpha * n_ * sampled_gradient_x

        # z1, z2 = y1, y2

        x1_list.append(x1)
        x2_list.append(x2)
        # if results.t_calls > 0 and results.la_mu_calls > 0:
        #     results.history_dual_gap.append(abs(oracle_stacker.oracle.prime(flows_averaged, corrs_averaged) + res_x))
        #     results.history_t_calls.append(results.t_calls)
        #     results.history_la_mu_calls.append(results.la_mu_calls)

    return results_t, results_la_mu
