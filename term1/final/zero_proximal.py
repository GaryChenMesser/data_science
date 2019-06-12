import numpy as np
from argparse import ArgumentParser

print("Initialize...")

beta_true      = np.zeros((200, 5))
beta_true[99]  = np.array([-1, 1, -1, 1, -1])
beta_true[199] = np.array([1, -1, 1, -1, 1])
beta_true      = beta_true.reshape(1000)

n = 200
X = np.random.normal(0, 1, (n, 1000))
tmp_y = np.exp(np.matmul(X, beta_true))
y = np.array([np.random.binomial(1, _y / (1 + _y)) for _y in tmp_y])

test_n = 10 * n
test_X = np.random.normal(0, 1, (test_n, 1000))
tmp_y = np.exp(np.matmul(test_X, beta_true))
test_y = np.array([np.random.binomial(1, _y / (1 + _y)) for _y in tmp_y])

tau = 0.0001
step = float(1 / (2 * np.sort(np.linalg.eig(np.matmul(X.T, X))[0])[-1]))
alpha = tau / step
alpha = [0.00001]
# 0.00001 -> (0.19)(0.2605)(0.00005)
tol = [0.0001, 0.00005, 0.000005]

records = [[] for i in range(len(alpha))]

print("Gradient Algorithm start.")

for r1, _alpha in enumerate(alpha):
    for r2, _tol in enumerate(tol):
        print("\n---------------------------------------------------------------------------")
        # Proximal Gradient Algorithm
        beta = np.array([np.ones((1000, )), np.zeros((1000, ))])
        z = np.zeros((1000, ))
        s = [1., 1.]
            
        for epoch in range(10000):
            if np.linalg.norm(beta[0] - beta[1]) <= _tol:
                break
        #while np.linalg.norm(beta[0] - beta[1]) > _tol:
            print("\r{}".format(np.linalg.norm(beta[0] - beta[1])), end='')
            beta[0] = beta[1]
            s[0] = s[1]
            X_exp = np.exp(np.matmul(X, z))
            
            # compute gradient of loss function
            gradient = 0
            for j in range(n):
                gradient += -y[j] * X[j] + X_exp[j] / (1 + X_exp[j]) * X[j]
        
            for i in range(1000):
                next = z[i] - step * gradient[i]
                if _alpha <= next * next / 2:
                    beta[1][i] = next
                else:
                    beta[1][i] = 0
            
            s[1] = (1 + np.sqrt(1 + 4 * s[0] * s[0])) / 2
            z = beta[1] + (s[0] - 1)/(s[1])*(beta[1] - beta[0])
            records[r1].append(np.linalg.norm(beta[0] - beta[1]))
        
        print("\ralpha = {}                        ".format(_alpha))
        print("tol = {}".format(_tol))
        print(beta[1][495:500])
        print(beta[1][995:1000])
        print(beta[1][490:495])
        print(beta[1][990:995])
        print(max(beta[1]))

        tmp_y = np.exp(np.matmul(X, beta[1]))
        trained_y = np.array([np.random.binomial(1, _y / (1 + _y)) for _y in tmp_y])

        #print(y)
        #print(trained_y)

        tmp_y = np.exp(np.matmul(test_X, beta[1]))
        pred_y = np.array([np.random.binomial(1, _y / (1 + _y)) for _y in tmp_y])

        print("Size of n: {}".format(n))
        print("Size of test_n: {}".format(test_n))
        print("MSE of beta: {}".format(np.linalg.norm(beta_true - beta[1])))
        print("Training error: {}".format(np.sum(np.abs(y - trained_y)) / n))
        print("Testing error: {}".format(np.sum(np.abs(test_y - pred_y)) / test_y.shape[0]))
with open("zero_fpg.out", 'w') as f:
    f.write(repr(alpha) + '\n' + repr(records))
