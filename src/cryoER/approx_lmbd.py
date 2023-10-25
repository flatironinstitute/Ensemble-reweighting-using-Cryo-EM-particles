import numpy as np
import torch
import argparse
import MDAnalysis as mda

import cryoER.imggen_torch as igt
from cryoER.tools import mdau_to_pos_arr

def circular_mask(n_pixels, radius):
    grid = torch.linspace(-.5*(n_pixels-1), .5*(n_pixels-1), n_pixels)
    grid_x, grid_y = torch.meshgrid(grid, grid, indexing='ij')
    r_2d = grid_x**2 + grid_y**2
    mask = r_2d < radius**2
    return mask

def signal_std_torch_batch(img):
    n_pixels = img.shape[1]
    radius = n_pixels*0.4
    mask = circular_mask(n_pixels, radius)
    image_masked = img[:, mask]
    signal_std = image_masked.pow(2).mean(1).sqrt()
    return signal_std

def _parse_args():

    parser = argparse.ArgumentParser(
        prog="cryoERMCMC",
        description="Perform MCMC sampling with Stan to sample posterior of \
            weights that reweights a conformational ensemble with cryo-EM \
            particles",
        # epilog = 'Text at the bottom of help'
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
        "-rm",
        "--rotmat_struc_imgstruc",
        type=str,
        default="rot_mats_struc_image.npy",
        help=".npy file for rotation matrix that aligns structure trajectory to image-generating trajectory",
    )
    parser.add_argument(
        "-o", "--outdir", default="./output/", help="directory for output files"
    )
    parser.add_argument(
        "-np",
        "--n_pixel",
        type=int,
        default=128,
        help="number of image pixels, use power of 2 for CTF purpose",
    )
    parser.add_argument(
        "-ps", "--pixel_size", type=float, default=0.2, help="pixel size in Angstrom"
    )
    parser.add_argument(
        "-sg", "--sigma", type=float, default=1.5, help="radius of Gaussian atom"
    )
    parser.add_argument(
        "-snr",
        "--signal_to_noise_ratio",
        type=float,
        default=1e-2,
        help="signal-to-noise ratio in synthetic images",
    )
    parser.add_argument(
        "-ctf",
        "--add_ctf",
        default=False,
        action="store_true",
        help="introduce CTF modulation (if True)",
    )
    parser.add_argument(
        "-dmin",
        "--defocus_min",
        type=float,
        default=0.027,
        help="Minimum defocus value in microns",
    )
    parser.add_argument(
        "-dmax",
        "--defocus_max",
        type=float,
        default=0.090,
        help="Maximum defocus value in microns",
    )
    parser.add_argument(
        "-dv",
        "--device",
        type=str,
        default="cpu",
        help='hardware device for calculation: "cuda" for GPU, or "cpu" for CPU',
    )
    parser.add_argument(
        "-nips",
        "--n_image_per_struc",
        type=int,
        default=100,
        help="number of images per structure for lambda approximation",
    )
    parser.add_argument(
        "-nb",
        "--n_batch",
        type=int,
        default=10,
        help="number of batch to separate the output files into for memory management",
    )
    return parser


def approx_lmbd(top_struc, traj_struc, n_pixel, pixel_size, sigma, signal_to_noise_ratio, add_ctf, defocus_min, defocus_max, n_image_per_struc, n_batch, device):

    uStruc = mda.Universe(top_struc, traj_struc)
    posStruc = mdau_to_pos_arr(uStruc)
    posStruc = posStruc.repeat(n_image_per_struc, 1, 1)

    _, _, images = igt.generate_images(
        posStruc, 
        n_pixel = n_pixel,
        pixel_size = pixel_size,
        sigma = sigma,
        snr = np.inf,
        rotation = True,
        add_ctf = add_ctf,
        defocus_min = defocus_min,
        defocus_max = defocus_max,
        batch_size = n_batch,
        device = device,
    )
    signal_std = signal_std_torch_batch(images)
    snr = signal_to_noise_ratio
    noise_std = signal_std / np.sqrt(snr)

    lmbd = noise_std.mean().numpy()
    print(snr, lmbd)

    return lmbd

if __name__ == "__main__":

    parser = _parse_args()
    args = parser.parse_args()
    _ = approx_lmbd(
        args.top_struc,
        args.traj_struc,
        args.n_pixel,
        args.pixel_size,
        args.sigma,
        args.signal_to_noise_ratio,
        args.add_ctf,
        args.defocus_min,
        args.defocus_max,
        args.n_image_per_struc,
        args.n_batch,
        args.device,
    )

    pass