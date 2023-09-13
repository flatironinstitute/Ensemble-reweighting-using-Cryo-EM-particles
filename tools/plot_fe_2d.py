# example script to plot 2D free energy surface with and without reweighting

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

cmap_fe = 'afmhot'

cv1 = np.loadtxt("../data/cv1.txt", dtype = float) # of each frame
cv2 = np.loadtxt("../data/cv2.txt", dtype = float)
cluster_labels = np.loadtxt("../data/cluster_labels.txt", dtype = int) # of each frame
weights_cluster = np.loadtxt("../src/output/weights_am.txt", dtype = float, usecols = 0)
weights = weights_cluster[cluster_labels]
n_bin = 100
xedges = np.linspace(0, 10, 101)
yedges = np.linspace(0, 10, 101)

H_init, xedges, yedges = np.histogram2d(cv1, cv2, bins=(xedges, yedges), density=True)
y_init = -np.log(H_init)
y_init -= np.amin(y_init)
y_init[np.isinf(y_init)] = 1e99

H_rewt, xedges, yedges = np.histogram2d(cv1, cv2, bins=(xedges, yedges), weights = weights, density=True)
y_rewt = -np.log(H_rewt)
y_rewt -= np.amin(y_rewt)
y_rewt[np.isinf(y_rewt)] = 1e99

xa = .5*(xedges[1:]+xedges[:-1])
ya = .5*(yedges[1:]+yedges[:-1])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot()
ax.set_position([0.14,0.12,0.85*0.8,0.85])
ai = ax.imshow(y_init, interpolation='none', origin='lower', cmap=cmap_fe, vmin=0, vmax=5,
extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

cax = fig.add_subplot()
cbar = fig.colorbar(ai, cax=cax, ax=ax)
cax.set_position([0.14+0.85*0.8+0.02,0.12,0.05,0.85])
cbar.set_label("Initial Free Energy",fontsize=30)

# ax.text(.02, .98, "Structure", fontsize=30, ha='left', va='top', transform=ax.transAxes)

ax.set_xticks(np.arange(0,11,2))
ax.set_yticks(np.arange(0,11,2))
ax.set_xlim(0,10)
ax.set_ylim(0,10)
ax.set_xlabel("CV1", fontsize=30)
ax.set_ylabel("CV2", fontsize=30)

plt.savefig("../src/output/fe_init.png", dpi=300)
plt.close()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot()
ax.set_position([0.14,0.12,0.85*0.8,0.85])
ai = ax.imshow(y_rewt, interpolation='none', origin='lower', cmap=cmap_fe, vmin=0, vmax=5,
extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

cax = fig.add_subplot()
cbar = fig.colorbar(ai, cax=cax, ax=ax)
cax.set_position([0.14+0.85*0.8+0.02,0.12,0.05,0.85])
cbar.set_label("Free Energy",fontsize=30)

# ax.text(.02, .98, "Structure", fontsize=30, ha='left', va='top', transform=ax.transAxes)

ax.set_xticks(np.arange(0,11,2))
ax.set_yticks(np.arange(0,11,2))
ax.set_xlim(0,10)
ax.set_ylim(0,10)
ax.set_xlabel("CV1", fontsize=30)
ax.set_ylabel("CV2", fontsize=30)

plt.savefig("../src/output/fe_rewt.png", dpi=300)
plt.close()


