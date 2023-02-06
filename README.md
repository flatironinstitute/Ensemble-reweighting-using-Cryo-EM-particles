# Ensemble reweighting using Cryo-EM particles

* Please cite: https://arxiv.org/abs/2212.05320
* Authors: Wai Shing Tang, David Silva-Sánchez, Julian Giraldo-Barreto, Bob Carpenter, Sonya Hanson, Alex H. Barnett, Erik H. Thiede, Pilar Cossio

## Bayesian inference framework for reweighting conformational ensembles using cryo-EM particles

Here, we present an ensemble refinement framework that estimates the ensemble density from a set of cryo-EM particles by reweighting a prior ensemble of conformations, e.g., from molecular dynamics simulations or structure prediction tools. 

Installation
-----------------
To install, run 
```
    pip install -e .
```
in this directory.

Running the Code
-----------------
We present an example for generating the synthetic images and calculating the structure-images distance matrix, starting from a set of conformations from MD
```
python3 -m cryoER.calc_image_struc_distance \
  --top_image ../data/image.pdb \
  --traj_image ../data/image.xtc \
  --top_struc ../data/struc.gro \
  --traj_struc ../data/struc_m10.xtc \
  --rotmat_struc_imgstruc ../data/rot_mats_struc_image.npy \
  --outdir ../data/ \
  --n_pixel 128 \
  --pixel_size 0.2 \
  --sigma 1.5 \
  --signal_to_noise_ratio 1e-2 \
  --add_ctf \
  --n_batch 1
```
To use GPU acceleration, add the arguments
```
  --device cuda
```
The output is a file with the structure-image distance matrix. 

To estimate the weights, run the MCMC script
```
python3 -m cryoER.run_cryoER_mcmc \
  --infileclustersize data/cluster_counts.txt \
  --infileimagedistance data/diff.npy \
  ---outdir output/ \
  --chains 4 \
  --iterwarmup 200 \
  --itersample 2000
```
If there is more than one input file of structure-image distance matrix, replace the argument accordingly:
```
  --infileimagedistance data/diff0.npy data/diff1.npy data/diff2.npy \
```
For parallelization, for example, running 4 chains in parallel with 15 threads per chain in a multithread settings, add the arguments
```
  --parallelchain 4 \
  --threadsperchain 15
```

The output is a file with the weight for each MD conformation. 
