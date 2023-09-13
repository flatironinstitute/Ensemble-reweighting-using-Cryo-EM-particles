# This script calculates the rotation matrices that aligns the image trajectories to the structure trajectories
# and return rotation matrices in .npy format, i.e. rot_mats_struc_image.npy

from tqdm import tqdm
import MDAnalysis as mda
import numpy as np
import torch

# This uses a faster torch SVD implementation from https://github.com/KinglittleQ/torch-batch-svd
from torch_batch_svd import svd 

cuda = torch.device('cuda')

print("Reading trajectory files...")

uImage = mda.Universe("../data/image.pdb", "../data/image.xtc")
uStruc = mda.Universe("../data/struc.pdb", "../data/struc_m10.xtc")

def mdau_to_pos_arr(u):
    protein_CA = u.select_atoms("protein and name CA")
    pos = torch.zeros((len(u.trajectory), len(protein_CA), 3), dtype=float)
    for i, ts in enumerate(u.trajectory):
        pos[i] = torch.from_numpy(protein_CA.positions)
    pos -= pos.mean(1).unsqueeze(1)
    return pos

posStruc = mdau_to_pos_arr(uStruc).cuda()
posImage = mdau_to_pos_arr(uImage).cuda()

n_batch = 5
batch_size = int(posStruc.shape[0]/n_batch)
while batch_size * n_batch < posStruc.shape[0]:
    batch_size += 1

rot_mats = torch.empty((posStruc.shape[0], posImage.shape[0], 3, 3), dtype=torch.float64, device="cpu")

print("Calculating rotation matrices...")
for i_batch in tqdm(range(n_batch)):
    batch_start  = i_batch*batch_size
    batch_end = (i_batch+1)*batch_size
    Hs = torch.einsum('nji,mjk->nmik', posImage, posStruc[batch_start:batch_end])
    u, s, v = svd(Hs.flatten(0,1))
    torch.cuda.empty_cache()
    R = torch.matmul(v, u.transpose(1,2))
    rot_mats[batch_start:batch_end] = R.cpu().reshape(Hs.shape).transpose(0,1)

print("Saving files...")
np.save("../data/rot_mats_struc_image.npy", rot_mats)

print("Complete!")
