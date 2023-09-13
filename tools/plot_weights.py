# example script to visualize the weights for reweighting

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 18

N_center = 10
filename_nm = '../data/cluster_counts.txt'
output_directory = '../src/output/'
nm = np.loadtxt(filename_nm)
nm /= np.sum(nm)
title = 'N_center = 10'

weights_mean, weights_std = np.loadtxt(output_directory+"weights_am.txt", unpack=True)
M = len(weights_mean)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.errorbar(np.arange(M), weights_mean, yerr=weights_std, fmt='o', label="amNm")
ax.set_xlabel("Model")
ax.set_ylabel("a_m")
ax.set_title(title)

ax.set_xticks(np.arange(0,M,5))
ax.set_yticks(np.arange(0,0.61,0.1))
ax.set_xlim(-0.5, M-0.5)
ax.set_ylim(0, 0.6)

plt.tight_layout()
# plt.savefig("%s/weights_am.png"%key, dpi=300)
plt.savefig(output_directory+"weights_am.png", dpi=300)
weights_mean, weights_std = np.loadtxt(output_directory+"weights_amnm.txt", unpack=True)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.plot(np.arange(M), nm, 'o', label="Nm")
ax.errorbar(np.arange(M), weights_mean, yerr=weights_std, fmt='o', label="amNm")
ax.set_xlabel("Model")
ax.set_ylabel("Weight")
ax.set_title(title)
ax.legend(
    loc='upper left',
    frameon=False,
)

ax.set_xticks(np.arange(0,M,5))
ax.set_yticks(np.arange(0,0.41,0.1))
ax.set_xlim(-0.5, M-0.5)
ax.set_ylim(0, 0.4)

plt.tight_layout()
plt.savefig(output_directory+"weights_amnm.png", dpi=300)
