#!/usr/bin/env python
import numpy as np
import os, sys
import csv
from scipy.special import logsumexp

N_center = 50
filename_nm = 'Test_Sep2023/nm.txt'
output_name = 'Test_NotUni'
output_directory = './output_%s/'%output_name
stan_directory = output_directory + 'stan_output/'

nm = np.loadtxt(filename_nm)
log_weights = []
lp = []
files = sorted(os.listdir(stan_directory))[-8:]
print(files)

nm /= np.sum(nm)
log_nm = np.log(nm)

log_weights_d = np.zeros((8,10000,N_center), dtype=float)
i = 0
for file in files:
    log_weights_chain = []
#    print(file)
    with open('%s/%s'%(stan_directory, file), newline='') as csvfile:
        reader = csv.DictReader(filter(lambda row: row[0]!='#', csvfile), )
        for row in reader:
            log_weights_row = [float(row["log_weights.%d"%i]) for i in range(1,N_center+1)]
            log_weights_chain.append(log_weights_row)

    log_weights = np.array(log_weights_chain)
    log_weights_d[i,:,:] = log_weights
    i += 1

# log_weights_d = log_weights_d[:,5000:,:]
log_am = log_weights_d - logsumexp(log_weights_d, axis=2)[:,:,None]

am = np.exp(log_am)
am_mean = am.mean((0,1))
am_std = am.std((0,1))

am_mean_std = np.vstack((am_mean, am_std)).T

np.savetxt(output_directory+"weights_am_LogLikeMat.txt", am_mean_std, fmt='%.6f')

log_amnm = log_weights_d + log_nm[None,None,:]
log_amnm -= logsumexp(log_amnm, axis=2)[:,:,None]

amnm = np.exp(log_amnm)
amnm_mean = amnm.mean((0,1))
amnm_std = amnm.std((0,1))

amnm_mean_std = np.vstack((amnm_mean, amnm_std)).T

np.savetxt(output_directory+"weights_amnm_LogLikeMat.txt", amnm_mean_std, fmt='%.6f')

