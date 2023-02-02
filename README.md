# cryoER

To run the MCMC scripts
```
python3 run_stan.py
  --infileclustersize data/cluster_counts.txt
  --infileimagedistance data/diff.npy
  ---outdir output/
  --chains 4
  --iterwarmup 200
  --itersample 2000 
```

For parallelization, for running 4 chains in parallel, 15 threads per chain in a multithread settings, add the arguments
```
...
  --parallelchain 4
  --threadsperchain 15
```
