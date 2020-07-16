import numpy as np

p = 20 / 38
alpha = 1 - p
C = 360

y = 20
num_trial = 1000
success = 0

for i in range(num_trial):
    print(i + 1, end='\r')
    tmp_y = 20
    k2 = 0
    while tmp_y > 0 and tmp_y < 200:
        time = -1.
        print(np.random.binomial(1, p))
        while(np.random.binomial(1, p) == 1):
            time += 0.5
        tmp_y += time
        
    if tmp_y >= 200:
        success += 1

print()
print(success)
prob = success / num_trial * np.power(alpha / p, 360)
print(prob)
