import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--mode", type=int)
parser.add_argument("--fast", type=int)
args = parser.parse_args()

print("Initialize...")

beta_true = np.array([-2, -2, 2, 2, -2])
p = 500

n = 10000
X = np.random.normal(0, 1, (n, p))
y = np.matmul(X[:, ::100], beta_true) + np.random.normal(0, np.sqrt(0.5), (n))

test_n = 10 * n
test_X = np.random.normal(0, 1, (test_n, p))
test_y = np.matmul(test_X[:, ::100], beta_true) + np.random.normal(0, np.sqrt(0.5), (test_n))

#tau = 0.0001
tau = 0.0001
step = 1 / (2 * np.sort(np.linalg.eig(np.matmul(X.T, X))[0])[-1])
alpha = tau / step

if args.mode == 0:
    fast = [args.fast]
    alpha = [alpha]
    
elif args.mode == 1:
    fast = [0, 1]
    alpha = [alpha]
    for i in range(4):
        alpha.append(alpha[-1] / 10)

records = [[[] for j in range(len(fast))] for i in range(len(alpha))]

print("Gradient Algorithm start.")

for r1, _alpha in enumerate(alpha):
    for r2, _fast in enumerate(fast):
        # Proximal Gradient Algorithm
        if _fast == 0:
            beta = np.array([np.ones((p, )), np.zeros((p, ))])
            
            #for epoch in range(500):
            while np.linalg.norm(beta[0] - beta[1]) > 0.000005:
                beta[0] = beta[1]
                # compute gradient of loss function
                grad_loss = np.matmul(y - np.matmul(X, beta[0]), -X)
        
                for i in range(p):
                    next = beta[0][i] - step * grad_loss[i]
                    if _alpha <= next * next / 2:
                        beta[1][i] = next
                    else:
                        beta[1][i] = 0
        
                records[r1][r2].append(np.linalg.norm(beta[0] - beta[1]))
        
        # Fast Proximal Gradient Algorithm
        else:
            beta = np.array([np.ones((p, )), np.zeros((p, ))])
            z = np.zeros((p, ))
            s = [1., 1.]
            
            #for epoch in range(500):
            while np.linalg.norm(beta[0] - beta[1]) > 0.000005:
                beta[0] = beta[1]
                s[0] = s[1]
                # compute gradient of loss function
                grad_loss = np.matmul(y - np.matmul(X, z), -X)
        
                for i in range(p):
                    next = z[i] - step * grad_loss[i]
                    if _alpha <= next * next / 2:
                        beta[1][i] = next
                    else:
                        beta[1][i] = 0
        
                s[1] = (1 + np.sqrt(1 + 4 * s[0] * s[0])) / 2
                z = beta[1] + (s[0] - 1)/(s[1])*(beta[1] - beta[0])
        
                records[r1][r2].append(np.linalg.norm(beta[0] - beta[1]))

for i in range(len(alpha)):
    alpha[i] *= step

with open("out1", 'w') as f:
    f.write(repr(alpha) + '\n' + repr(records))

beta_true      = np.zeros((500,))
beta_true[0] = -2
beta_true[100] = -2
beta_true[200] = 2
beta_true[300] = 2
beta_true[400] = -2
pred_y = np.matmul(test_X[:], beta[1])

print("Size of n: {}".format(n))
print("Size of test_n: {}".format(test_n))
print("MSE of beta: {}".format(np.linalg.norm(beta_true - beta[1])))
print("Training error: {}".format(np.linalg.norm(y - np.matmul(X, beta[1])) / n))
print("Testing error: {}".format(np.linalg.norm(test_y - pred_y) / test_y.shape[0]))
