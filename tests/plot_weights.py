import numpy as np
from matplotlib import pyplot as plt

import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 18

N_center = 50
filename_nm = 'Test_Sep2023/nm.txt'
output_name = 'Test_NotUni'
output_directory = './output_%s/'%output_name

key = "Test_uniform_random_lik_1e6_s1_N100000"
title = key
nm = np.loadtxt("%s/nm.txt"%key)
weights_mean, weights_std = np.loadtxt("%s/weights_am.txt"%key, unpack=True)

# output_directory = "./Test_Sep2023/realikeli_script/"
# nm = np.loadtxt("Test_Sep2023/nm.txt")
# nm /= np.sum(nm)
# weights_mean, weights_std = np.loadtxt("Test_Sep2023/realikeli_script/weights_am_LogLikeMat.txt", unpack=True)
# title = "LogLikeMat_realikeli_script"

M = len(weights_mean)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.errorbar(np.arange(M), weights_mean, yerr=weights_std, fmt='o', label="amNm")
ax.set_xlabel("Model")
ax.set_ylabel("a_m")
ax.set_title(title)
ax.set_xticks(np.arange(0,M,5))
# ax.set_yticks(np.arange(0,0.21,0.05))
ax.set_xlim(-0.5, M-0.5)
# ax.set_ylim(0, 0.4)
plt.tight_layout()
plt.savefig("%s/weights_am.png"%key, dpi=300)
# plt.savefig("Test_Sep2023/realikeli_script/weights_am_LogLikeMat.png", dpi=300)

weights_mean, weights_std = np.loadtxt("%s/weights_amnm.txt"%key, unpack=True)
# weights_mean, weights_std = np.loadtxt("Test_Sep2023/realikeli_script/weights_amnm_LogLikeMat.txt", unpack=True)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.plot(np.arange(M)+0.1, nm, 'o', label="Nm")
ax.errorbar(np.arange(M)-0.1, weights_mean, yerr=weights_std, fmt='o', label="amNm")
ax.set_xlabel("Model")
ax.set_ylabel("Weight")
ax.set_title(title)
ax.legend(
    loc='upper left',
    frameon=False,
)

ax.set_xticks(np.arange(0,M,5))
ax.set_yticks(np.arange(0,0.21,0.05))

ax.set_xlim(-0.5, M-0.5)
ax.set_ylim(0, 0.2)
plt.tight_layout()
plt.savefig("%s/weights_amnm.png"%key, dpi=300)
# plt.savefig("Test_Sep2023/realikeli_script/weights_amnm_LogLikeMat.png", dpi=300)

