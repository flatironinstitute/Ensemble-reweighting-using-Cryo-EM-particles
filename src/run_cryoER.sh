# Bash script for running Ensemble reweighting on example data files

# generating the synthetic images and calculating the structure-images distance matrix, starting from a set of conformations from MD
python3 -m cryoER.calc_image_struc_distance \
  --top_image ../data/image.pdb \
  --traj_image ../data/image.xtc \
  --top_struc ../data/struc.pdb \
  --traj_struc ../data/struc_m10.xtc \
  --rotmat_struc_imgstruc ../data/rot_mats_struc_image.npy \
  --outdir ../data/ \
  --n_pixel 128 \
  --pixel_size 0.2 \
  --sigma 1.5 \
  --signal_to_noise_ratio 1.0 \
  --add_ctf \
  --n_batch 1

# run the MCMC script to estimate the weights
python3 -m cryoER.run_cryoER_mcmc \
  --infileclustersize ../data/cluster_counts.txt \
  --infileimagedistance ../data/diff_npix128_ps0.20_s1.5_snr1.0E-02.npy \
  --outdir output/ \
  --chains 4 \
  --iterwarmup 200 \
  --itersample 2000 \
  --parallelchain 4 \
  --threadsperchain 8