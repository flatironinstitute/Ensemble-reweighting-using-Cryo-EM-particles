#!/usr/bin/env python
import numpy as np
import os, sys
import csv
from scipy.special import logsumexp

N_center = 50
#dataset = 'r1'
directory = "./stan_output"#.format(dataset)
log_weights = []
lp = []
files = sorted(os.listdir(directory))[-8:]

log_weights_d = np.zeros((8,10000,N_center), dtype=float)
i = 0
for file in files:
    log_weights_chain = []
#    print(file)
    with open('%s/%s'%(directory,file), newline='') as csvfile:
        reader = csv.DictReader(filter(lambda row: row[0]!='#', csvfile), )
        for row in reader:
            log_weights_row = [float(row["log_weights.%d"%i]) for i in range(1,N_center+1)]
            log_weights_chain.append(log_weights_row)

    log_weights = np.array(log_weights_chain)
    log_weights_d[i,:,:] = log_weights
    i += 1

log_weights_d -= logsumexp(log_weights_d, axis=2)[:,:,None]

weights_d = np.exp(log_weights_d)
weights_d_mean = weights_d.mean((0,1))
weights_d_std = weights_d.std((0,1))

np.savetxt("weights.txt", weights_d_mean, fmt='%.6f')
