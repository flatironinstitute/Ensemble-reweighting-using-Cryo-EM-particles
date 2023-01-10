import os, sys
from cmdstanpy import cmdstan_path, CmdStanModel
import numpy as np
import json

N_center = 20 ## Number of centers, i.e., representative configuration from MD
filelabels = [
    "nonoise", ## Image with no noise added
    "snr0", ## Image with noise added, with SNR = 1e0 = 1
    "snr1", ## SNR = 1e-1 = 0.1
    "snr2", ##  ...
    "snr3",
    "snr4",
]
## lambda in exponential term in the likelihood in Eq. 10 and 17, to represent colorless noise standard deviation
## this is a known parameter from the respective image noise generation process for the synthetic images, for image with no noise, lambda is arbitrary chosen to be 1e-6
lmbds = [
    1e-6,
    7.2e-4,
    2.3e-3,
    7.2e-3,
    2.3e-2,
    7.2e-2,
]

## Compile Stan script
my_stanfile = os.path.join('.', 'cryo-er.stan')
my_model = CmdStanModel(stan_file=my_stanfile,
    cpp_options={"STAN_THREADS": True, 
    },
    # compile='force',
)
print(my_model.exe_info())

## Specifying filenames
dataset = "replicakm%dtodesres"%N_center
output_directory = "/mnt/home/wtang/ceph/cryoER_stan/" 
original_directory = "/mnt/home/wtang/ceph/cryo_ensemble_refinement/data/"
kd_directory = "/mnt/home/wtang/Code/cryo_ensemble_refinement/src/"

for filelabel, lmbd in zip(filelabels, lmbds):

    json_filename = "%s/dmat/Dmat_A01_b1_%s_%s_output_directory.json"%(output_directory,dataset,filelabel)
    thislmbd_output_directory = "%s/output/Dmat_A01_b1_%s_%s_output_directory_8_10000"%(output_directory,dataset,filelabel)
    try:
        os.mkdir(thislmbd_output_directory)
    except FileExistsError:
        pass
    print(N_center, dataset, filelabel, lmbd)

    ## Read N_m, the number of conformations that are in the mth cluster
    labels_kd = np.loadtxt(kd_directory+"km_labels_120k_%d.txt"%N_center).astype(int)
    values, counts = np.unique(labels_kd, return_counts=True)

    counts = counts.astype(float)
    counts /= np.sum(counts)
    log_Nm = np.log(counts)

    ## Read distance matrix between cryoEM images and MD structures
    distance = np.load(original_directory+'diff_A01_b1_%s_%s_batch0.npy'%(dataset,filelabel))
    for i_batch in range(1,10):
        data = np.load(original_directory+'diff_A01_b1_%s_%s_batch%d.npy'%(dataset,filelabel,i_batch))
        distance = np.hstack((distance,data))

    ## Write json files for reading into Stan program
    M = distance.shape[0]
    N = distance.shape[1]

    norm = .5/(lmbd**2)
    Dmat = -norm*distance.T

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

    ## Run Stan model to perform MCMC sampling on posterior in Eq. 10 and 17
    data_file = os.path.join('.', json_filename)
    fit = my_model.sample(data=data_file,
        chains=8,
        sig_figs=12,
        parallel_chains=8,
        threads_per_chain=15,
        iter_warmup=1000,
        iter_sampling=10000,
        show_console=True,
    )
    
    ## Save Stan output, i.e., posterior samples, in CSV format, in a specified folder
    fit.save_csvfiles(dir=thislmbd_output_directory)

