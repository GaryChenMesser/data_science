import numpy as np

p = 18 / 38
alpha = 1 - p

y = 20
num_trial = 1000
success = 0

for i in range(num_trial):
    print(i + 1, end='\r')
    tmp_y = 20
    while tmp_y != 0 and tmp_y != 200:
        tmp_y += np.random.binomial(1, alpha) * 2 - 1
    
    if tmp_y == 200:
        success += 1

prob = success / num_trial * np.power(p / alpha, 180)
print(prob)
print(success)
