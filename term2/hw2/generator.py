import numpy as np

data = []
data.append(map(str, map(float, np.random.binomial(1, 0.1, size=200) + 1)))
data.append(map(str, map(float, np.random.binomial(1, 0.9, size=200) + 1)))
data.append(map(str, map(float, np.random.binomial(1, 0.1, size=200) + 1)))
data.append(map(str, map(float, np.random.binomial(1, 0.9, size=200) + 1)))
data.append(map(str, map(float, np.random.binomial(1, 0.1, size=200) + 1)))
data.append(map(str, map(float, np.random.binomial(1, 0.9, size=200) + 1)))

with open('test1.txt', 'w') as f:
    for i in range(len(data)):
        for index, j in enumerate(data[i]):
            f.write(j)
            if i != len(data) - 1 or index < len(data[i]) - 1:
                f.write('   ')
