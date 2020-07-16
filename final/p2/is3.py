import numpy as np

dim = 1

def f_of_x(x):
    """
    This is the main function we want to integrate over.
    Args:
    - x (float) : input to function; must be in radians
    Return:
    - output of function f(x) (float)
    """
    return np.sin(np.sqrt(x)) * np.exp(-100 * x)

# this is the template of our weight function g(x)
def g_of_x(x, A, lamda):
    tmp = 1
    for _x in x:
        tmp *= A * np.exp(-lamda*_x)
    return tmp

def inverse_G_of_r(r, lamda):
    return -np.log(r) / lamda

def get_IS_variance(lamda, num_samples):
    """
    This function calculates the variance if a Monte Carlo
    using importance sampling.
    Args:
    - lamda (float) : lamdba value of g(x) being tested
    Return: 
    - Variance
    """
    A = lamda
    int_max = 1
    
    # get sum of squares
    running_total = 0
    for i in range(num_samples):
        x = np.random.uniform(0., int_max, dim)
        running_total += (f_of_x(x)/g_of_x(x, A, lamda))**2
        #print("run: ", running_total)
    
    sum_of_sqs = running_total / num_samples
    
    # get squared average
    running_total = 0
    for i in range(num_samples):
        x = np.random.uniform(0., int_max, dim)
        running_total += f_of_x(x)/g_of_x(x, A, lamda)
        #print("run: ", running_total)
    sq_ave = (running_total/num_samples)**2
    
    
    return sum_of_sqs - sq_ave

# get variance as a function of lambda by testing many
# different lambdas

test_lamdas = [i*0.05 for i in range(1, 61)]
variances = []
'''
for i, lamda in enumerate(test_lamdas):
    print("lambda {}/{}: {}".format(i+1, len(test_lamdas), lamda))
    A = lamda
    variances.append(get_IS_variance(lamda, 10000))
    print(variances[-1])
 
optimal_lamda = test_lamdas[np.argmin(np.asarray(variances))]
IS_variance = variances[np.argmin(np.asarray(variances))]

print("Optimal Lambda: {}".format(optimal_lamda))
print("Optimal Variance: {}".format(IS_variance))
print("Error: {}".format((IS_variance/10000)**0.5))
'''
def importance_sampling_MC(lamda, num_samples):
    A = lamda
    
    running_total = 0
    for i in range(num_samples):
        r = np.random.uniform(0.,1., dim)
        running_total += f_of_x(inverse_G_of_r(r, lamda=lamda))/g_of_x(inverse_G_of_r(r, lamda=lamda), A, lamda)
        #print("	imp: ", running_total)
    approximation = float(running_total/num_samples)
    return approximation

# run simulation
num_samples = 30000
optimal_lamda = 30.
approx = importance_sampling_MC(optimal_lamda, num_samples)
variance = get_IS_variance(optimal_lamda, num_samples)
error = (variance/num_samples)**0.5

# display results
print("Importance Sampling Approximation: {}".format(approx))
print("Variance: {}".format(variance))
print("Error: {}".format(error))
