import numpy as np


def d_i_j(lambda_l_i: np.ndarray, lambda_w_j: np.ndarray, gammaT_ij: np.ndarray) -> np.ndarray:
    return np.exp(-(1 + gammaT_ij + lambda_w_j + lambda_l_i[:, np.newaxis]))


def primal_f(d_ij, gammaT_ij):
    return (d_ij * gammaT_ij).sum() + (d_ij * np.log(d_ij)).sum()


def dual_f(lambda_l_i, lambda_w_j, gammaT_ij, L_i, W_j):
    return -(
        (lambda_l_i * L_i).sum() + (lambda_w_j * W_j).sum() + d_i_j(lambda_l_i, lambda_w_j, gammaT_ij).sum()
    )


def dual_gap(d_ij, lambda_l_i, lambda_w_j, gammaT_ij, L_i, W_j):
    primal = primal_f(d_ij, gammaT_ij)
    dual = dual_f(lambda_l_i, lambda_w_j, gammaT_ij, L_i, W_j)  # = -varphi
    return primal - dual
