import os, sys
from cmdstanpy import cmdstan_path, CmdStanModel
from scipy.special import logsumexp
import numpy as np
import json

import multiprocessing
print(multiprocessing.cpu_count())

filename_nm = 'Test_Sep2023/nm.txt'
filename_loglik = 'Test_Sep2023/NotUni-LogLikeMat'
output_name = 'Test_NotUni'

Nm = np.loadtxt(filename_nm)
Nm /= np.sum(Nm)
log_lik_mat = np.loadtxt(filename_loglik)
log_lik_mat = np.array(log_lik_mat)

print(Nm.shape)
print(log_lik_mat.shape)

out_directory = './output_%s/'%output_name
try:
    os.mkdir(out_directory)
except:
    pass
stan_output = out_directory + 'stan_output/'
try:
    os.mkdir(stan_output)
except:
    pass
json_filename = out_directory + 'stan_input.json'

M = log_lik_mat.shape[1]    # Number of model
N = log_lik_mat.shape[0]    # Number of images

log_Nm = np.log(Nm)

Dmat = log_lik_mat

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

my_stanfile = os.path.join('.', 'cryo-er.stan')
# my_stanfile = os.path.join('.', 'cryo-er.stan')
my_model = CmdStanModel(stan_file=my_stanfile,
    cpp_options={"STAN_THREADS": True, 
        # "STAN_MPI": True, 
        # "CXX": "mpicxx",
        # "TBB_CXX_TYPE": "gcc",
    },
    # compile='force',
)
print(my_model.exe_info())

fit = my_model.sample(data=json_filename,
    chains=8,
    sig_figs=12,
    parallel_chains=8,
    threads_per_chain=4,
    iter_warmup=1000,
    iter_sampling=10000,
    show_console=True,
)

fit.save_csvfiles(dir=stan_output)