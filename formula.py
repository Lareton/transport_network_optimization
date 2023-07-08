import numpy as np

nodes_cnt = 100
la = np.random.rand(nodes_cnt)
mu = np.random.rand(nodes_cnt)
l = np.random.rand(nodes_cnt)
w = np.random.rand(nodes_cnt)
T = np.random.rand(nodes_cnt, nodes_cnt)
t = np.random.rand(nodes_cnt * 3 // 2)
sigma_star = lambda t: t
gamma = 1

def calc(T, la, mu, l, w, t, sigma_star, gamma):
    func1 = gamma * np.log(np.sum(np.exp((np.sum(-T + la[None,...], axis=1) + mu) / gamma))) - l @ la - w @ mu + np.sum(sigma_star(t))
    return func1

print(calc(T, la, mu, l, w, t, sigma_star, gamma))
