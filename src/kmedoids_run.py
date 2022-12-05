import kmedoids
import numpy as np

distances = np.load("/mnt/home/wtang/ceph/cryo_ensemble_refinement/data/replica/distance_CA_120k_square.npy")
c = kmedoids.fasterpam(distances, 5)

labels = c.labels
medoids = c.medoids
np.savetxt("km_labels_120k_5.txt", labels, fmt="%d")
np.savetxt("km_medoids_120k_5.txt", medoids, fmt="%d")

print(c)
