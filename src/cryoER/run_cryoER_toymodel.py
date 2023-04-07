import os, sys
from cmdstanpy import cmdstan_path, CmdStanModel
import numpy as np
import json
import multiprocessing
print(multiprocessing.cpu_count())

my_stanfile = os.path.join('.', 'cryo-er.stan')
my_model = CmdStanModel(stan_file=my_stanfile,
    cpp_options={"STAN_THREADS": True, 
        # "STAN_MPI": True, 
        # "CXX": "mpicxx",
        # "TBB_CXX_TYPE": "gcc",
    },
    # compile='force',
)

print(my_model.exe_info())

dataset = "toy_model"
output_directory = "../output"
filelabel = "sigma1"

filename = "%s/dmat/Dmat_%s_%s.json"%(output_directory,dataset,filelabel)
directory = "%s//Dmat_%s_%s"%(output_directory,dataset,filelabel)
try:
    os.mkdir(directory)
except FileExistsError:
    pass

## 3D Toy model
## Generate Gaussian distributed data points as "images"
np.random.seed(0)
means = np.array([
    [0.0,0.0,0.0],
    [2.0,4.0,-2.0],
    [4.0,0.0,0.0],
])  # Center of the Gaussians
covariance = np.array([
    [1.0,0.0,0.0],
    [0.0,1.0,0.0],
    [0.0,0.0,1.0],
])  # Width of the Gaussians
data = np.array([])
sizes = [5000, 3000, 2000]  # relative population of the 3 Gaussians in the data set
for mean, size in zip(means, sizes):
    cluster = np.random.multivariate_normal(mean=mean,cov=covariance,size=size)
    if data.size: 
        data = np.vstack((data,cluster))
    else:
        data = cluster

sample = np.array([
    [0.0,0.0,0.0],   # point A
    # [-0.1,0.0,0.0],    # point A1
    # [0.1,0.0,0.0],     # point A2
    [2.0,4.0,-2.0],    # point B
    # [3.0,2.0,-1.0],  # point D
    [4.0,0.0,0.0],     # point C
])
distance = np.sum((sample[None,:] - data[:,None])**2,axis=-1)

M = distance.shape[1]
N = distance.shape[0]
N_pix = 3
sigma = 1.0
norm = .5/(sigma**2)
Dmat = -norm*distance

dictionary = {
    "M": M,
    "N": N,
    "Dmat": [list(a) for a in Dmat]
}

json_object = json.dumps(dictionary, indent=4)
with open(filename, "w") as f:
    f.write(json_object)
    f.close()

data_file = os.path.join('.', filename)
fit = my_model.sample(data=data_file,
    chains=8,
    sig_figs=12,
    parallel_chains=8,
    threads_per_chain=20,
    iter_warmup=1000,
    iter_sampling=10000,
)

fit.save_csvfiles(dir=directory)

