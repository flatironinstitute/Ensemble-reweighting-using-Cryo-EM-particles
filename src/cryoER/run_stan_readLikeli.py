import os, sys
#from cmdstanpy import cmdstan_path, CmdStanModel
from cmdstanpy import CmdStanModel
import importlib.resources
from scipy.special import logsumexp
import numpy as np
import json

import multiprocessing
print(multiprocessing.cpu_count())

out_directory = './stan_output/'
module_path = importlib.resources.files("cryoER")
path = os.path.join(module_path, "cryo-er.stan")

key = 'LogL'
json_filename = os.path.join('.', '{}.json'.format(key))

# list files in directory
negloglik = []
#filename = os.path.join('LogLikeMat', file)
data = np.loadtxt('LogLikeMat')
negloglik = np.array(data).T

M = negloglik.shape[0]    # Number of model
N = negloglik.shape[1]    # Number of images = 

print("Struct-Images ", M, N)

# if want to read Nm input
Nm = np.loadtxt("nm.txt", dtype=float)
Nm /= np.sum(Nm)
log_Nm = np.log(Nm)

#uniform Nm
#Nm = np.ones(M)/M
#log_Nm = np.log(Nm)

# norm = .5/(sigma**2)
# Dmat = -norm*distance.T
Dmat = -negloglik.T

dictionary = {
    "M": M,
    "N": N,
    "logNm": list(log_Nm),
    "Dmat": [list(a) for a in Dmat]
}

json_object = json.dumps(dictionary, indent=4)

with open(json_filename, "w") as f:
    f.write(json_object)
    f.close()

my_stanfile = os.path.join(module_path, 'cryo-er.stan')
# my_stanfile = os.path.join('.', 'cryo-er.stan')
my_model = CmdStanModel(stan_file=my_stanfile,
    cpp_options={"STAN_THREADS": True, 
        # "STAN_MPI": True, 
        # "CXX": "mpicxx",
        # "TBB_CXX_TYPE": "gcc",
    },
    # compile='force',
)
# print(my_model.exe_info())

# for dataset in datasets:
# N_center = int(sys.argv[1])
# dataset = "replicakm%dtodesres"%N_center

fit = my_model.sample(data=json_filename,
    chains=8,
    sig_figs=12,
    parallel_chains=8,
    threads_per_chain=15,
    iter_warmup=1000,
    iter_sampling=10000,
    show_console=True,
)

fit.save_csvfiles(dir=out_directory)
