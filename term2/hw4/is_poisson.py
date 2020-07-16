import numpy as np
from scipy.stats import poisson

p = 0.95
alpha = 0.99

y = 20
num_trial = 1
success = 0.
count  = 0

for i in range(num_trial):
    print(i + 1, end='\r')
    tmp_y = 20
    k = 0
    while tmp_y > 0 and tmp_y < 200:
        _k = np.random.poisson(alpha) - 1
        k += _k
        tmp_y += _k
    
    if tmp_y >= 200:
        success += np.exp(alpha - p) * np.power(p / alpha, k)
        count += 1

prob = success / num_trial
print(prob)
print(count)

prob = poisson.sf(18, p)
print(prob)
