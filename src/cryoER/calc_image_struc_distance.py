import numpy as np
from tqdm import tqdm 
import torch
import imggen_torch as igt
import os, sys
import MDAnalysis as mda
import argparse

parser = argparse.ArgumentParser(
    prog = 'cryoERMCMC',
    description = 'Perform MCMC sampling with Stan to sample posterior of weights that reweights a conformational ensemble with cryo-EM particles',
    # epilog = 'Text at the bottom of help'
)

######## Input parameters ########

parser.add_argument('-ip', '--top_image', type=str, default="image.gro",
    help="topology file for image-generating trajectory")
parser.add_argument('-it', '--traj_image', type=str, default="image.xtc",
    help="trajectory file for image-generating trajectory")
parser.add_argument('-sp', '--top_struc', type=str, default="struc.gro",
    help="topology file for structure trajectory")
parser.add_argument('-st', '--traj_struc', type=str, default="struc.xtc",
    help="trajectory file for structure trajectory")
parser.add_argument('-rm', '--rotmat_struc_imgstruc', type=str, default="rot_mats_struc_image.npy",
    help=".npy file for rotation matrix that aligns structure trajectory to image-generating trajectory")
parser.add_argument('-o', '--outdir', default="./output/",
    help="directory for output files")
parser.add_argument('-np', '--n_pixel', type=int, default=128,
    help="number of image pixels, use power of 2 for CTF purpose")
parser.add_argument('-ps', '--pixel_size', type=float, default=0.2,
    help="pixel size in Angstrom")
parser.add_argument('-sg', '--sigma', type=float, default=1.5,
    help="radius of Gaussian atom")
parser.add_argument('-snr', '--signal_to_noise_ratio', type=float, default=1e-2,
    help="signal-to-noise ratio in synthetic images")
parser.add_argument('-ctf', '--add_ctf', default=False, action='store_true',
    help="introduce CTF modulation (if True)")
parser.add_argument('-dv', '--device', type=str, default="cpu",
    help="hardware device for calculation: \"cuda\" for GPU, or \"cpu\" for CPU")
parser.add_argument('-nb', '--n_batch', type=int, default=10,
    help="number of batch to separate the output files into for memory management")

######## ######## ######## ########

args = parser.parse_args()
top_image = args.top_image
traj_image = args.traj_image
top_struc = args.top_struc
traj_struc = args.traj_struc
rotmat_struc_imgstruc = args.rotmat_struc_imgstruc

outdir = args.outdir
try:
    os.mkdir(outdir)
except FileExistsError:
    pass

n_pixel = args.n_pixel
pixel_size = args.pixel_size
sigma = args.sigma
snr = args.signal_to_noise_ratio
ctf_bool = args.add_ctf
device = args.device
n_batch = args.n_batch

######## ######## ######## ########

file_prefix = "npix%d_ps%.2f_s%.1f_snr%.1E"%(n_pixel, pixel_size, sigma, snr)

def mdau_to_pos_arr(u):
    protein_CA = u.select_atoms("protein and name CA")
    pos = torch.zeros((len(u.trajectory), len(protein_CA), 3), dtype=float)
    for i, ts in enumerate(u.trajectory):
        pos[i] = torch.from_numpy(protein_CA.positions)
    pos -= pos.mean(1).unsqueeze(1)
    return pos

print("Reading trajectory...")

## Reading image trajectory
uImg = mda.Universe(
    top_image,
    traj_image
)
coord_img = mdau_to_pos_arr(uImg)

## Read structure centers
uStr = mda.Universe(
    top_struc,
    traj_struc
)
coord_str = mdau_to_pos_arr(uStr)

N_center = len(uStr.trajectory) ## Number of centers, i.e., representative configuration from MD

rot_mats_align = torch.from_numpy(np.load(rotmat_struc_imgstruc))

n_struc = coord_str.shape[0]

batch_start = 0
n_frame = len(uImg.trajectory)

if n_batch == 1:
    
    pos_batch = coord_img
    if device == "cuda":
        pos_batch = pos_batch.cuda()
    
    rot_mats, ctfs, images = igt.generate_images(
            pos_batch,
            n_pixel = n_pixel,  ## use power of 2 for CTF purpose
            pixel_size = pixel_size,
            sigma = sigma,
            snr = snr,
            add_ctf = ctf_bool,
            batch_size = 16,
            device = device,
    )

    np.save('%s/rot_mats_%s.npy'%(outdir,file_prefix), rot_mats)
    np.save('%s/ctf_%s.npy'%(outdir,file_prefix), ctfs)
    np.save('%s/images_%s.npy'%(outdir,file_prefix), images)

    rot_mats = torch.from_numpy(np.load('%s/rot_mats_%s.npy'%(outdir,file_prefix)))
    ctfs = torch.from_numpy(np.load('%s/ctf_%s.npy'%(outdir,file_prefix)))
    images = torch.from_numpy(np.load('%s/images_%s.npy'%(outdir,file_prefix)))
    
    n_image = images.shape[0]

    # if not os.path.exists('/mnt/home/wtang/ceph/cryo_ensemble_refinement/data/diff_%s_%s_batch%d.npy'%(dataset, snrlabel, i_batch)):

    diff = np.zeros((n_struc, n_image), dtype=float)
    for i in tqdm(range(n_struc)):
        aligned_coord = coord_str[i].unsqueeze(0).matmul(rot_mats_align[i]).matmul(rot_mats)
        if device == "cuda":
            aligned_coord = aligned_coord.cuda()
        diff[i] = igt.calc_struc_image_diff(
            aligned_coord,
            n_pixel = n_pixel,  ## use power of 2 for CTF purpose
            pixel_size = pixel_size, 
            sigma = sigma, 
            images = images,
            ctfs = ctfs,
            batch_size = 16,
            device = device,
        )

    print("Saving...")
    np.save('%s/diff_%s.npy'%(outdir, file_prefix), diff)

else:

    batch_size = int(n_frame/n_batch)
    while batch_size*n_batch < n_frame:
        batch_size += 1

    for i_batch in range(n_batch):

        print("Batch %d"%i_batch)
        pos_batch = coord_img[i_batch*batch_size:(i_batch+1)*batch_size]
        if device == "cuda":
            pos_batch = pos_batch.cuda()
        rot_mats, ctfs, images = igt.generate_images(
                pos_batch,
                n_pixel = n_pixel,  ## use power of 2 for CTF purpose
                pixel_size = pixel_size,
                sigma = sigma,
                snr = snr,
                add_ctf = ctf_bool,
                batch_size = 16,
                device = device,
        )

        np.save('%s/rot_mats_%s_batch%d.npy'%(outdir,file_prefix,i_batch), rot_mats)
        np.save('%s/ctf_%s_batch%d.npy'%(outdir,file_prefix,i_batch), ctfs)
        np.save('%s/images_%s_batch%d.npy'%(outdir,file_prefix,i_batch), images)

        print("Batch %d..."%i_batch)

        rot_mats = torch.from_numpy(np.load('%s/rot_mats_%s_batch%d.npy'%(outdir,file_prefix,i_batch)))
        ctfs = torch.from_numpy(np.load('%s/ctf_%s_batch%d.npy'%(outdir,file_prefix,i_batch)))
        images = torch.from_numpy(np.load('%s/images_%s_batch%d.npy'%(outdir,file_prefix,i_batch)))
        
        n_image = images.shape[0]
        batch_end = batch_start + n_image

        # if not os.path.exists('/mnt/home/wtang/ceph/cryo_ensemble_refinement/data/diff_%s_%s_batch%d.npy'%(dataset, snrlabel, i_batch)):

        diff = np.zeros((n_struc, n_image), dtype=float)
        for i in tqdm(range(n_struc)):
            aligned_coord = coord_str[i].unsqueeze(0).matmul(rot_mats_align[i,batch_start:batch_end]).matmul(rot_mats)
            if device == "cuda":
                aligned_coord = pos_batch.cuda()
            diff[i] = igt.calc_struc_image_diff(
                aligned_coord,
                n_pixel = n_pixel,  ## use power of 2 for CTF purpose
                pixel_size = pixel_size, 
                sigma = sigma, 
                images = images,
                ctfs = ctfs,
                batch_size = 16,
                device = device,
            )

        print("Saving...")
        np.save('%s/diff_%s_batch%d.npy'%(outdir, file_prefix, i_batch), diff)

        batch_start = batch_end