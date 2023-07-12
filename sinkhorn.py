from typing import Tuple, Union

import numba
import numpy as np
from numba import njit
from scipy.special import logsumexp

from oracle_utils import primal_f, d_i_j

np.set_printoptions(suppress=True)


@njit(parallel=True, fastmath=True)
def numba_logsumexp_stable(arr: np.ndarray) -> np.ndarray:
    """
    Equivalent to scipy.special.logsumexp(arr, axis=1) for 2-D array
    Taken from https://github.com/aesara-devs/aesara/issues/404#issuecomment-840417025
    :param arr: 2-D array
    :return: logsumexp, summing along axis=1
    """

    n, m = arr.shape
    assert arr.ndim == 2
    out = np.zeros(n)

    for i in numba.prange(n):
        p_max = np.max(arr[i])
        res = 0
        for j in range(m):
            res += np.exp(arr[i, j] - p_max)
        res = np.log(res) + p_max
        out[i] = res

    return out


class Sinkhorn:
    """Finds trip distribution (correspondences matrix) for ONE demand layer. If there are more than one demand layer,
    they should be processed independently by different instances of this class.
    TODO: remove call of _make_nonzero, and implement interface described below. This would require dropping user type
     axis from sinkhorn, because arrivals/departures for different user type can have different shapes
     (number of nonzero elements before dropping zeros) and therefore are cant be numpy represented as numpy arrays.
     Passing user types to sinkhorn has been required when arrivals wasnt restricted by used types
    Takes nonzero departures and arrivals as input, centroids with zero demand/supply should be dropped. The
    resulting correspondence matrices are rectangular and should be filled with zero rows/columns outside this class"""

    def __init__(
        self,
        departures: np.ndarray,
        arrivals: np.ndarray,
        max_iter: int,
        eps=1e-6,
        crit_check_period=10,
        use_numba=False,
    ):
        """
        :param departures: nonzero departures[zone_ind] for current demand layer
        :param arrivals: nonzero arrivals[zone_ind] for current demand layer
        :param max_iter:
        :param eps: absolute tolerance for function value and constraints violation norm
        :param crit_check_period: number of iterations to check criteria at each
        """
        self.L_i = departures  # departures
        self.W_j = arrivals  # arrivals
        self.n_types = self.L_i.shape[0]
        self.max_iter = max_iter
        self.eps = eps
        self.crit_check_period = crit_check_period
        self.use_numba = use_numba

    def _sinkhorn_iteration(
        self, k: int, gammaT_ij: np.ndarray, lambda_w_j: np.ndarray, lambda_l_i: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param k: iteration number
        :param gammaT_tij:  cost matrices[user type][zone_from][zone_to] (already multiplied by gamma)
        :param lambda_w_tj: dual variables for arrival constraints for each user type
        :param lambda_l_ti: dual variables for arrival constraints for each user type
        :return: dual variables
        """
        # TODO: choose automatically by matrix size (and magnitude of linear part maybe)
        if self.use_numba:
            return self._numba_iteration(k, gammaT_ij, lambda_w_j, lambda_l_i)
        else:
            return self._scipy_iteration(k, gammaT_ij, lambda_w_j, lambda_l_i)

    def _numba_iteration(
        self,
        k: int,
        gammaT_ij: np.ndarray,
        lambda_w_j: np.ndarray,
        lambda_l_i: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if k % 2 == 0:
            # print(lambda_w_tj.shape, gammaT_tij.shape, self.L_ti.shape)
            lambda_l_i = numba_logsumexp_stable(-lambda_w_j[np.newaxis, :] - 1 - gammaT_ij) - np.log(self.L_i)
        else:
            lambda_w_j = numba_logsumexp_stable((-lambda_l_i[:, np.newaxis] - 1 - gammaT_ij).T) - np.log(self.W_j) 

        return lambda_w_j, lambda_l_i

    def _scipy_iteration(
        self,
        k: int,
        gammaT_ij: np.ndarray,
        lambda_w_j: np.ndarray,
        lambda_l_i: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if k % 2 == 0:
            lambda_l_i = logsumexp((-lambda_w_j[np.newaxis, :] - 1 - gammaT_ij), b=1 / self.L_i[:, np.newaxis], axis=1)
        else:
            lambda_w_j = logsumexp((-lambda_l_i[:, np.newaxis] - 1 - gammaT_ij), b=1 / self.W_j[np.newaxis, :], axis=0)

        return lambda_w_j, lambda_l_i

    def run(
        self,
        gammaT_ij: np.ndarray,
        lambda_l_i: Union[np.ndarray, None] = None,
        lambda_w_j: Union[np.ndarray, None] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        :param gammaT_ij: cost matrix (already multiplied by gamma)
        :param lambda_l_i: initial lambda_l_i (starting point)
        :param lambda_w_j: initial lambda_w_j (starting point)
        :return: corr matrices for each user type and dual variables
        """
        # gammaT[gammaT == 0.0] = 100.0
        if lambda_l_i is None:
            lambda_l_i = np.zeros(self.L_i.shape)
        if lambda_w_j is None:
            lambda_w_j = np.zeros(self.W_j.shape)

        k = 0
        # TODO: consider not to check crit at each iteration to speedup
        while True:
            if k > 0 and not k % self.crit_check_period:
                if self._criteria(lambda_l_i, lambda_w_j, gammaT_ij):
                    break

            lambda_w_j, lambda_l_i = self._sinkhorn_iteration(k, gammaT_ij, lambda_w_j, lambda_l_i)

            k += 1
            if k == self.max_iter:
                raise RuntimeError("Max iter exceeded in Sinkhorn")

        print(f"sink iters: {k}")

        return d_i_j(lambda_l_i, lambda_w_j, gammaT_ij), lambda_l_i, lambda_w_j

    def _criteria(self, lambda_l_i: np.ndarray, lambda_w_j: np.ndarray, gammaT_ij: np.ndarray) -> bool:
        d_ij = d_i_j(lambda_l_i, lambda_w_j, gammaT_ij)
        grad_l = d_ij.sum(axis=1) - self.L_i
        grad_w = d_ij.sum(axis=0) - self.W_j
        dual_grad = np.hstack((grad_l, grad_w))

        dual_grad_norm = np.linalg.norm(dual_grad)  # equal to constraints violation norm
        inner_prod = -np.hstack((lambda_l_i, lambda_w_j)) @ dual_grad  # upper bound for f(x_k) - f(x^*)
        print(f"Sinkhorn crit contraints norm: {dual_grad_norm}, dual gap: {inner_prod}, eps: {self.eps}")
        print(f"primal val: {primal_f(d_ij, gammaT_ij)}")

        return dual_grad_norm < self.eps and inner_prod < self.eps
