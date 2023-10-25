# example script to calculate the mean weights and standard deviation from MCMC samples (Stan output)

import numpy as np
import os, sys
import csv
from scipy.special import logsumexp

N_center = 10
filename_nm = '../data/cluster_counts.txt'
output_directory = '../src/output/'
stan_directory = output_directory + 'Stan_output/'

nm = np.loadtxt(filename_nm)
log_weights = []
lp = []
files = sorted(os.listdir(stan_directory))
print(files)

nm /= np.sum(nm)
log_nm = np.log(nm)

log_weights_d = []
i = 0
for file in files:
    log_weights_chain = []
    with open('%s/%s'%(stan_directory, file), newline='') as csvfile:
        reader = csv.DictReader(filter(lambda row: row[0]!='#', csvfile), )
        for row in reader:
            log_weights_row = [float(row["log_weights.%d"%i]) for i in range(1,N_center+1)]
            log_weights_chain.append(log_weights_row)
    log_weights = np.array(log_weights_chain)
    log_weights_d.append(log_weights)

log_weights_d = np.array(log_weights_d)

log_am = log_weights_d - logsumexp(log_weights_d, axis=2)[:,:,None]

am = np.exp(log_am)
am_mean = am.mean((0,1))
am_std = am.std((0,1))

am_mean_std = np.vstack((am_mean, am_std)).T

np.savetxt(output_directory+"weights_am.txt", am_mean_std, fmt='%.6f')

log_amnm = log_weights_d + log_nm[None,None,:]
log_amnm -= logsumexp(log_amnm, axis=2)[:,:,None]

amnm = np.exp(log_amnm)
amnm_mean = amnm.mean((0,1))
amnm_std = amnm.std((0,1))

amnm_mean_std = np.vstack((amnm_mean, amnm_std)).T

np.savetxt(output_directory+"weights_amnm.txt", amnm_mean_std, fmt='%.6f')

