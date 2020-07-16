import numpy as np
from scipy.stats import norm
import sys

with open(sys.argv[1], 'r') as f:
    X = np.array(list(map(float, f.read().split('   '))))

iteration = 10000
n_cluster = 2
Y         = np.random.randint(n_cluster, size=X.shape[0])

for i in range(iteration):
    print(i+1, end='\r')
    p1 = np.sum(Y) / X.shape[0]
    p2 = np.sum(1 - Y) / X.shape[0]
    mu1 = np.sum(Y * X) / np.sum(Y)
    mu2 = np.sum((1 - Y) * X) / np.sum(1 - Y)
    st1 = np.sqrt(np.sum(Y * (X - mu1) ** 2) / np.sum(Y))
    st2 = np.sqrt(np.sum((1 - Y) * (X - mu2) ** 2) / np.sum(1 - Y))
    
    N1 = norm(mu1, st1).pdf(X)
    N2 = norm(mu2, st2).pdf(X)
    Y = p1 * N1 / (p1 * N1 + p2 * N2)
    
    '''
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
        loss0 = g[0].pdf(data[j])
        loss1 = g[1].pdf(data[j])
        cluster[j] = np.argmax([loss0, loss1])
        #cluster[j] = np.random.binomial(1, loss1 / (loss0 + loss1))
        #if i == 19:
        #    print(loss0, loss1, cluster[j], data[j])
    '''
    
print("mean = {}, {}\n".format(mu1, mu2))
print("std = {}, {}\n".format(st1, st2))
print("number = {}, {}\n".format(p1, p2))
