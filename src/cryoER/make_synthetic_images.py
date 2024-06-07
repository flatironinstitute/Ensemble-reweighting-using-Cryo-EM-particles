import numpy as np
from tqdm import tqdm
import torch
import os
import MDAnalysis as mda
import argparse

from cryoER.tools import mdau_to_pos_arr
import cryoER.imggen_torch as igt

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
        "-nb",
        "--n_batch",
        type=int,
        default=10,
        help="number of batches to separate the output files into for memory management",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=10,
        help="sizes of batches to separate the output files into for memory management",
    )
    return parser


######## ######## ######## ########


def make_synthetic_images(
    top_image = "image.gro",
    traj_image = "image.xtc",
    n_pixel = 128,
    pixel_size = 0.2,
    sigma = 1.5,
    snr = 1e-2,
    n_image_per_struc = 1,
    add_ctf = False,
    defocus_min = 0.027,
    defocus_max = 0.090,
    device = "cpu",
    n_batch = None,
    batch_size = 16,
    outdir = None
):

    ######## ######## ######## ########

    file_prefix = "npix%d_ps%.2f_s%.1f_snr%.1E" % (n_pixel, pixel_size, sigma, snr)
    print("Reading trajectory...")

    ## Reading image trajectory
    print("Reading image trajectory from %s and %s..." % (top_image, traj_image))
    uImg = mda.Universe(top_image, traj_image)
    coord_img = mdau_to_pos_arr(uImg)
    if n_image_per_struc > 1:
        coord_img = coord_img.repeat(n_image_per_struc, 1, 1)

    if n_batch is not None and batch_size is None:
        n_frames = coord_img.shape[0]
        batch_size = n_frames//n_batch

    coord_img = coord_img.to(device)
    rot_mats, ctfs, images = igt.generate_images(
        coord_img,
        n_pixel=n_pixel,  ## use power of 2 for CTF purpose
        pixel_size=pixel_size,
        sigma=sigma,
        snr=snr,
        add_ctf=add_ctf,
        defocus_min=defocus_min,
        defocus_max=defocus_max,
        batch_size=batch_size,
        device=device,
    )
    if outdir is not None:
        print("Saving images to %s..." % outdir)
        np.save("%s/rot_mats_%s.npy" % (outdir, file_prefix), rot_mats.cpu().numpy())
        np.save("%s/ctf_%s.npy" % (outdir, file_prefix), ctfs.cpu().numpy())
        np.save("%s/images_%s.npy" % (outdir, file_prefix), images.cpu().numpy())
        print("Done!")

    return rot_mats, ctfs, images

if __name__ == "__main__":

    parser = _parse_args()
    args = parser.parse_args()

    top_image = args.top_image
    traj_image = args.traj_image
    outdir = args.outdir
    n_pixel = args.n_pixel
    pixel_size = args.pixel_size
    sigma = args.sigma
    snr = args.signal_to_noise_ratio
    add_ctf = args.add_ctf
    defocus_min = args.defocus_min
    defocus_max = args.defocus_max
    device = args.device
    n_batch = args.n_batch
    batch_size = args.batch_size

    _, _, _ = make_synthetic_images(
        top_image = top_image,
        traj_image = traj_image,
        n_pixel = n_pixel,
        pixel_size = pixel_size,
        sigma = sigma,
        snr = snr,
        add_ctf = add_ctf,
        defocus_min = defocus_min,
        defocus_max = defocus_max,
        device = device,
        n_batch = n_batch,
        batch_size = batch_size,
        outdir = outdir
    )
    
    pass
