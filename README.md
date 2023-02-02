# Ensemble reweighting using Cryo-EM particles


To generate the synthetic images and calculate the structure-images distanace matrix
```
python3 calc_image_struc_distance.py \
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


To run the MCMC scripts
```
python3 run_stan.py \
  --infileclustersize data/cluster_counts.txt \
  --infileimagedistance data/diff.npy \
  ---outdir output/ \
  --chains 4 \
  --iterwarmup 200 \
  --itersample 2000
```
If there are more than one input file of structure-image distance matrix, replace the argument accordingly:
```
  --infileimagedistance data/diff0.npy data/diff1.npy data/diff2.npy \
```
For parallelization, for running 4 chains in parallel, 15 threads per chain in a multithread settings, add the arguments
```
  --parallelchain 4 \
  --threadsperchain 15
```
