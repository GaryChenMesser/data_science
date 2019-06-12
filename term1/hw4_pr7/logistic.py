import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# ----------------Initialization---------------
# beta_true
beta_true = np.array([-1, 1, -1, 1, -1, 1, -1, 1, -1, 1])

# Observation
X = np.random.normal(0, 1, (200, 10))
for i in range(200):
    X[i][0] = 1

# Labels
X_exp = np.exp(np.matmul(X, beta_true))
Y = np.random.binomial(1, X_exp / (1 + X_exp))

# Benchmark
clf = LogisticRegression(tol=0.000005, max_iter = 100000, solver="liblinear", C = 10000).fit(X, Y)
f = np.matmul(X, np.array(clf.coef_)[0])
X_exp = np.exp(f)
loss_star = -np.matmul(Y, f) + np.sum(np.log(np.ones(200) + X_exp))

# Stop criterion
tol = 0.000005
max_iter = 10000

# Step size
gram = np.zeros((10, 10))
for i in range(200):
    gram += np.matmul(X[i].reshape(10, 1), X[i].reshape(1, 10))
step_size = 4 / max(np.linalg.eig(gram)[0])

# Start condition 
beta = np.zeros(10)
X_exp = np.exp(np.matmul(X, beta))
gradient = 1 / step_size
one = np.ones(200)

# Record
G = []
L = []

for i in range(max_iter):
    if np.max(gradient) < tol:
        break
    
    gradient = 0
    for j in range(200):
        gradient += -Y[j] * X[j] + X_exp[j] / (1 + X_exp[j]) * X[j]
    
    beta -= step_size * gradient
    f = np.matmul(X, beta)
    X_exp = np.exp(f)
    
    loss = -np.matmul(Y, f) + np.sum(np.log(one + X_exp))
    
    G.append(np.linalg.norm(gradient))
    L.append(loss - loss_star)

print(beta)

plt.subplot(121)
plt.plot(G)
plt.yscale("log")
plt.title("Norm of Gradient")
plt.subplot(122)
plt.plot(L)
plt.yscale("log")
plt.title("Difference of loss between Optimizer")
plt.show()
