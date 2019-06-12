import numpy as np
from scipy.stats import norm
import sys

with open(sys.argv[1], 'r') as f:
    data = np.array(map(float, f.read().split('   ')))

iteration = 20
n_cluster = 2
cluster   = np.random.randint(n_cluster, size=data.shape[0])

for i in range(iteration):
    # Expectation
    mu = [0. for j in range(n_cluster)]
    st = [0. for j in range(n_cluster)]
    for k in range(n_cluster):
        tmp = [data[j] for j in range(data.shape[0]) if cluster[j] == k]
        mu[k] = np.mean(tmp)
        st[k] = np.std(tmp)
    
    # Maximization
    g = [norm(mu[k], st[k]) for k in range(n_cluster)]
    
    for j in range(data.shape[0]):
        cluster[j] = np.argmax([g[k].pdf(data[j]) for k in range(n_cluster)])
    
print("mean = {}, std = {}\n".format(mu, st))
count = [0, 0]
for j in range(data.shape[0]):
   count[cluster[j]] += 1
print("number = {}\n".format(count))
