import numpy as np

N_center = 10
kd_directory = "/mnt/home/wtang/Code/cryo_ensemble_refinement/src/"
labels_kd = np.loadtxt(kd_directory+"km_labels_120k_%d.txt"%N_center).astype(int)
values, counts = np.unique(labels_kd, return_counts=True)
for i in range(len(values)):
    print(values[i], counts[i])
np.savetxt("../data/cluster_counts.txt", counts, fmt="%d")