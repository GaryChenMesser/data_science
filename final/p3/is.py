import numpy as np

p = (18/38)
alpha = 1 - p

y = 20
num_trial = 1000
success = 0

for i in range(num_trial):
    print(i + 1, end='\r')
    tmp_y = y
    while tmp_y > 0 and tmp_y < 21:
        if np.random.binomial(1, p) == 1:
            tmp_y += .5
        else:
            tmp_y -= 1.
    
    if tmp_y >= 21:
        success += 1

prob = success / num_trial
print(prob)
print(success)
