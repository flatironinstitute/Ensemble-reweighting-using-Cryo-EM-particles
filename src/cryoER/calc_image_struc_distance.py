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
        "-i",
        "--images",
        type=str,
        default=None,
        help=".npy file for images",
    )
    parser.add_argument(
        "-c",
        "--ctfs",
        type=str,
        default=None,
        help=".npy file for CTFs",
    )
    parser.add_argument(
        "-ri",
        "--rot_mats_image",
        type=str,
        default=None,
        help=".npy file for rotation matrix that specifies the orientation of each image",
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
        "-nb",
        "--batch_size",
        type=int,
        default=10,
        help="number of batch to separate the output files into for memory management",
    )
    return parser


######## ######## ######## ########


def calc_image_struc_distance(
    images = None,
    ctfs = None,
    rot_mats_image = None,
    top_struc = "struc.gro",
    traj_struc = "struc.xtc",
    rotmat_struc_imgstruc = None,
    outdir = "./output/",
    n_pixel = 128,
    pixel_size = 0.2,
    sigma = 1.5,
    snr = 1e-2,
    add_ctf = False,
    defocus_min = 0.027,
    defocus_max = 0.090,
    device = "cpu",
    batch_size = 16
):

    ######## ######## ######## ########

    file_prefix = "npix%d_ps%.2f_s%.1f_snr%.1E" % (n_pixel, pixel_size, sigma, snr)

    print("Reading trajectory from %s and %s..." % (top_struc, traj_struc))
    uStr = mda.Universe(top_struc, traj_struc)
    coord_str = mdau_to_pos_arr(uStr)
    n_struc = coord_str.shape[0]

    if rotmat_struc_imgstruc is not None:
        print("Reading struc-images alignment matrices from %s..." % rotmat_struc_imgstruc)
        rot_mats_align = torch.from_numpy(np.load(rotmat_struc_imgstruc)).to(device)
    else:
        print("No struc-imgstruc alignment matrices specified, assume only using poses")

    if images is None:
        print("Loading images from %s/images_%s.npy..." % (outdir, file_prefix))
        images = np.load("%s/images_%s.npy" % (outdir, file_prefix))
    if ctfs is None:
        print("Loading CTFs from %s/ctf_%s.npy..." % (outdir, file_prefix))
        ctfs = np.load("%s/ctf_%s.npy" % (outdir, file_prefix))
    if rot_mats_image is None:
        print("Loading poses from %s/rot_mats_%s.npy..." % (outdir, file_prefix))
        rot_mats_image = np.load("%s/rot_mats_%s.npy" % (outdir, file_prefix))

    if not torch.is_tensor(images):
        images = torch.from_numpy(images)
    if not torch.is_tensor(ctfs):
        ctfs = torch.from_numpy(ctfs)
    if not torch.is_tensor(rot_mats_image):
        rot_mats_image = torch.from_numpy(rot_mats_image)

    images = images.to(device)
    ctfs = ctfs.to(device)
    rot_mats_image = rot_mats_image.to(device)

    n_image = images.shape[0]
    diff = np.zeros((n_struc, n_image), dtype=float)
    for i in tqdm(range(n_struc), desc="Computing image-structure distance for structure"):
        if rotmat_struc_imgstruc is not None:
            aligned_coord = coord_str[i].unsqueeze(0).matmul(rot_mats_align[i]).matmul(rot_mats_image)
        else:
            aligned_coord = coord_str[i].unsqueeze(0).matmul(rot_mats_image)
        diff[i] = igt.calc_struc_image_diff(
            aligned_coord,
            n_pixel=n_pixel,  ## use power of 2 for CTF purpose
            pixel_size=pixel_size,
            sigma=sigma,
            images=images,
            ctfs=ctfs,
            batch_size=batch_size,
            device=device,
        )

    print("Saving...")
    np.save("%s/diff_%s.npy" % (outdir, file_prefix), diff)
    print("Done!")

    return diff

if __name__ == "__main__":

    parser = _parse_args()
    args = parser.parse_args()

    if args.images is not None:
        print("Loading images from %s..." % args.images)
        images = np.load(args.images)
    if args.ctfs is not None:
        print("Loading CTFs from %s..." % args.ctfs)
        ctfs = np.load(args.ctfs)
    if args.rot_mats_image is not None:
        print("Loading poses from %s..." % args.rot_mats_image)
        rot_mats_image = np.load(args.rot_mats_image)
    top_struc = args.top_struc
    traj_struc = args.traj_struc
    rotmat_struc_imgstruc = args.rotmat_struc_imgstruc
    outdir = args.outdir
    n_pixel = args.n_pixel
    pixel_size = args.pixel_size
    sigma = args.sigma
    snr = args.signal_to_noise_ratio
    add_ctf = args.add_ctf
    defocus_min = args.defocus_min
    defocus_max = args.defocus_max
    device = args.device
    batch_size = args.batch_size

    _ = calc_image_struc_distance(
        images = images,
        ctfs = ctfs,
        rot_mats_image = rot_mats_image,
        top_struc = top_struc,
        traj_struc = traj_struc,
        rotmat_struc_imgstruc = rotmat_struc_imgstruc,
        outdir = outdir,
        n_pixel = n_pixel,
        pixel_size = pixel_size,
        sigma = sigma,
        snr = snr,
        add_ctf = add_ctf,
        defocus_min = defocus_min,
        defocus_max = defocus_max,
        device = device,
        batch_size = batch_size
    )
    
    pass
