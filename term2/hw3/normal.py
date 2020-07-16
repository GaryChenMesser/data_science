import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def importance_sampling(num_simulations, num_samples, verbose=True, plot=False):
    
    if verbose:
        print("===================================================")
        print("{} Importance Sampling Simulations of size {}".format(num_simulations, num_samples))
        print("===================================================\n")
    
    num_simulations = int(num_simulations)
    num_samples = int(num_samples)
    
    probas = []
    for i in range(num_simulations):
        mu_1, sigma_1 = 0, 0.01
        mu_2, sigma_2 = 0, 0.03
        mu_1_n, sigma_1_n = 0, 0.02
        mu_2_n, sigma_2_n = 0, 0.06
        
        # setup pdfs
        old_pdf_1 = norm(mu_1, sigma_1)
        new_pdf_1 = norm(mu_1_n, sigma_1_n)
        old_pdf_2 = norm(mu_2, sigma_2)
        new_pdf_2 = norm(mu_2_n, sigma_2_n)

        length = np.random.normal(mu_1_n, sigma_1_n, num_samples)
        voltage = np.random.normal(mu_2_n, sigma_2_n, num_samples)
        
        # calculate current
        num = 50 * np.square((0.6 - voltage))
        denum = 0.1 + length
        I = num / denum
        
        # calculate f
        true_condition = np.where(I >= 275)

        # calculate weight
        num = old_pdf_1.pdf(length) * old_pdf_2.pdf(voltage)
        denum = new_pdf_1.pdf(length) * new_pdf_2.pdf(voltage)
        weights = num / denum

        # select weights for nonzero f
        weights = weights[true_condition]

        # compute unbiased proba
        proba = np.sum(weights) / num_samples
        probas.append(proba)
        
        false_condition = np.where(I < 275)
        if plot:
            if i == num_simulations -1:
                plt.scatter(length[true_condition], voltage[true_condition], color='r')
                plt.scatter(length[false_condition], voltage[false_condition], color='b')
                plt.xlabel(r'$\Delta L$ [$\mu$m]')
                plt.ylabel(r'$\Delta V_{TH}$ [V]')
                plt.title("Monte Carlo Estimation of P(I > 275)")
                plt.grid(True)
                plt.savefig(plot_dir + 'imp_sampling_{}.pdf'.format(num_samples), format='pdf', dpi=300)
                plt.show()
    
    
    mean_proba = np.mean(probas)
    std_proba = np.std(probas)
    
    if verbose:
        print("Probability Mean: {}".format(mean_proba))
        print("Probability Std: {}".format(std_proba))
    
    return probas

probas = importance_sampling(10, 10000, plot=False)
