import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy.stats import norm
import sys

def like(pro, epi, z, x):
    return z[0] * (z[1] * np.log(1 - epi[1]) + (1 - z[1]) * np.log(epi[1])) +\
           (1 - z[0]) * (z[1] * np.log(epi[0]) + (1 - z[1]) * np.log(1 - epi[0])) +\
           z[1] * (x * np.log(pro[1]) + (1 - x) * np.log(1 - pro[1])) +\
           (1 - z[1]) * (x * np.log(pro[0]) + (1 - x) * np.log(1 - pro[0]))

with open(sys.argv[1], 'r') as f:
    data = np.array(list(map(float, f.read().split('   '))), dtype = int) - 1


iteration = 10
n_cluster = 2
z         = np.zeros(data.shape, dtype = int)

print(data.shape)
print(z.shape)

for i in range(iteration):
    # Expectation
    if i == 0:
        pro = [0.8, 0.2]
        epi = [0.1, 0.1]
    else:
        for k in range(n_cluster):
            tmp = [data[j] for j in range(data.shape[0]) if z[j] == k]
            pro[k] = np.mean(tmp)
            tmp = [abs(z[j + 1] - z[j]) for j in range(data.shape[0] - 1) if z[j] == k]
            epi[k] = np.mean(tmp)
    
    #epi[1] /= 4
    #epi[0] /= 4
    
    print("probability = {}, epsilon = {}\n".format(pro, epi))
    
    # Maximization
    accu = [data[0] * np.log(1 - pro[0]) + (1 - data[0]) * np.log(pro[0]), \
            data[0] * np.log(pro[1]) + (1 - data[0]) * np.log(1 - pro[1])]
    #accu = [0., 0.]
    #print(accu)
    last = [[0, 0] for j in range(data.shape[0] - 1)]
    for j in range(data.shape[0] - 1):
        tmp = [0., 0.]
        for k in range(2):
            #print(k)
            #print(like(pro, epi, [0, k], data[j+1]) + accu[0])
            #print(like(pro, epi, [1, k], data[j+1]) + accu[1])
            if like(pro, epi, [0, k], data[j+1]) + accu[0] < \
               like(pro, epi, [1, k], data[j+1]) + accu[1]:
                last[j][k] = 1
                tmp[k] = like(pro, epi, [1, k], data[j+1]) + accu[1]
            else:
                last[j][k] = 0
                tmp[k] = like(pro, epi, [0, k], data[j+1]) + accu[0]
        
        accu = tmp
    
    if accu[1] > accu[0]:
        z[-1] = 1
    else:
        z[-1] = 0
    
    z[-1] = 1
    
    for j in reversed(range(data.shape[0] - 1)):
        z[j] = last[j][z[j + 1]]
    
    #print(last)
    #print(z)            
    #print("***probability = {}, epsilon = {}\n".format(pro, epi))
    #count = [0, 0]
    #for j in range(data.shape[0]):
    #   count[cluster[j]] += 1
    #print("number = {}\n".format(count))
print(z)
