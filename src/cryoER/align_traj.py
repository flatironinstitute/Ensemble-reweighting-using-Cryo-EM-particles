import torch
import numpy as np
import argparse
import MDAnalysis as mda
from tqdm import tqdm

from cryoER.tools import mdau_to_pos_arr

def _parse_args():
    parser = argparse.ArgumentParser(
        prog="cryoERMCMC",
        description="Perform MCMC sampling with Stan to sample posterior of \
            weights that reweights a conformational ensemble with cryo-EM \
            particles",
        # epilog = 'Text at the bottom of help'
    )
    parser.add_argument(
        "-ip",
        "--top_image",
        type=str,
        default="image.gro",
        help="topology file for image-generating trajectory",
    )
    parser.add_argument(
        "-it",
        "--traj_image",
        type=str,
        default="image.xtc",
        help="trajectory file for image-generating trajectory",
    )
    parser.add_argument(
        "-sp",
        "--top_struc",
        type=str,
        default="struc.gro",
        help="topology file for structure trajectory",
    )
    parser.add_argument(
        "-st",
        "--traj_struc",
        type=str,
        default="struc.xtc",
        help="trajectory file for structure trajectory",
    )
    parser.add_argument(
        "-o", "--outdir", default="./output/", help="directory for output files"
    )
    parser.add_argument(
        "-dv",
        "--device",
        type=str,
        default="cpu",
        help='hardware device for calculation: "cuda" for GPU, or "cpu" for CPU',
    )
    return parser


######## ######## ######## ########


def align_traj(
    top_image="image.gro",
    traj_image="image.xtc",
    top_struc="struc.gro",
    traj_struc="struc.xtc",
    outdir="./output/",
    device="cpu",
):
    
    uImage = mda.Universe(top_image, traj_image)
    uStruc = mda.Universe(top_struc, traj_struc)
    posImage = mdau_to_pos_arr(uImage)
    posStruc = mdau_to_pos_arr(uStruc)
    nImage = posImage.shape[0]
    nStruc = posStruc.shape[0]

    output_directory = outdir
    device = torch.device(device)

    rot_mats = torch.empty((nStruc, nImage, 3, 3), dtype=torch.float64, device="cpu")
    posImage = posImage.to(device)
    posStruc = posStruc.to(device)

    n_batch = 1
    batch_size = nStruc // n_batch
    print("Calculating rotation matrices...")
    for i_batch in tqdm(range(n_batch)):
        batch_start  = i_batch*batch_size
        batch_end = (i_batch+1)*batch_size
        Hs = torch.einsum('nji,mjk->nmik', posImage, posStruc[batch_start:batch_end])
        u, s, vh = torch.linalg.svd(Hs.flatten(0,1))
        v = vh.transpose(1,2)
        R = torch.matmul(v, u.transpose(1,2))
        rot_mats[batch_start:batch_end, :, :, :] = torch.reshape(R.cpu(), Hs.shape).transpose(0,1)

    rot_mats = rot_mats.cpu().numpy()
    np.save(output_directory + "rot_mats_struc_image.npy", rot_mats)

    print("Done!")

    return rot_mats

if __name__ == "__main__":

    parser = _parse_args()
    args = parser.parse_args()

    _ = align_traj(
        args.top_image,
        args.traj_image,
        args.top_struc,
        args.traj_struc,
        args.outdir,
        args.device,
    )

    pass