import mdtraj as md
import numpy as np
from tqdm import tqdm

## Calculate self distance matrix of structure trajectory for Clustering
t = md.load("/mnt/home/wtang/ceph/test_MD/chignolin/replicas/data/xtc/rmsfit_md_run_all_pbc_center_skip10.xtc",
    top="/mnt/home/wtang/ceph/test_MD/chignolin/gromacs_2022/data/gro/md_run_pbc_center.gro")
atom_indices = [a.index for a in t.topology.atoms if a.name == 'CA']

distances = np.empty((t.n_frames, t.n_frames))
for i in tqdm(range(t.n_frames)):
    distances[i] = md.rmsd(t, t, i, atom_indices=atom_indices, parallel=True)
print('Max pairwise rmsd: %f nm' % np.max(distances))
np.save("/mnt/home/wtang/ceph/cryo_ensemble_refinement/data/replica/distance_CA_120k_square.npy", distances)

## Calculate distance matrix between image trajectory and structures for ground truth result
frames = np.loadtxt("km_medoids_120k_3.txt").astype(int)
tref = md.load("/mnt/home/wtang/ceph/test_MD/chignolin/replicas/data/xtc/rmsfit_md_run_all_pbc_center_skip10.xtc",
    top="/mnt/home/wtang/ceph/test_MD/chignolin/gromacs_2022/data/gro/md_run_pbc_center.gro")
ref_atom_indices = [a.index for a in tref.topology.atoms if a.name == 'CA']
t = md.load("trj/rmsfit_md_run_desres_skip5.xtc",
    top="/mnt/home/wtang/ceph/desres_data/DESRES-Trajectory_CLN025-0-protein/CLN025-0-protein.pdb")
atom_indices = [a.index for a in t.topology.atoms if a.name == 'CA']

distances = np.empty((len(frames), t.n_frames))
for i, f in tqdm(enumerate(frames)):
    distances[i] = md.rmsd(t, tref, f, atom_indices=atom_indices, ref_atom_indices=ref_atom_indices, parallel=True)
print('Max pairwise rmsd: %f nm' % np.max(distances))
np.save("/mnt/home/wtang/ceph/cryo_ensemble_refinement/data/replica/distance_CA_km3_desres.npy", distances)
