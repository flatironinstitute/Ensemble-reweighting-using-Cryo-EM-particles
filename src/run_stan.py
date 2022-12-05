import os, sys
from cmdstanpy import cmdstan_path, CmdStanModel
import numpy as np
import json
import multiprocessing
print(multiprocessing.cpu_count())

filelabels = [
    "nonoise",
    "snr0",  ## SNR = 1e-0
    "snr1",  ## SNR = 1e-1
    "snr2",
    "snr3",
    "snr4",
    "snr5",
    "snr6",
    "snr7",
]
sigmas = [
    1e-6,
    1.6e-3,
    5e-3,
    1.6e-2,
    5e-2,
    1.6e-1,
    5e-1,
    1.6,
    5,
]

my_stanfile = os.path.join('.', 'cryo-bife-log-wtf.stan')
my_model = CmdStanModel(stan_file=my_stanfile,
    cpp_options={"STAN_THREADS": True, 
    },
    # compile='force',
)
print(my_model.exe_info())

N_center = int(sys.argv[1])
dataset = "replicagmm%dtodesres"%N_center

ceph_directory = "/mnt/home/wtang/ceph/cryoER_stan/"

for filelabel, sigma in zip(filelabels, sigmas):

    ### Read outputs from cryo-EM imaging model

    filename = "%s/dmat/Dmat_%s_%s_wtf.json"%(ceph_directory,dataset,filelabel)
    directory = "%s/output/Dmat_%s_%s_wtf_8_10000"%(ceph_directory,dataset,filelabel)
    try:
        os.mkdir(directory)
    except FileExistsError:
        pass
    print(dataset, filelabel, sigma)

    original_directory = "/mnt/home/wtang/ceph/cryo_ensemble_refinement/data/"
    
    ## K-medoids clustering
    # labels_kd = np.loadtxt("/mnt/home/wtang/Code/cryo_ensemble_refinement/src/km_labels_120k_%d.txt"%N_center).astype(int)
    # values, counts = np.unique(labels_kd, return_counts=True)

    ## GMM clustering
    # labels_gm = np.load("/mnt/home/wtang/Code/gmm_clustering/output/gmm_clusters_%d.npy"%N_center).astype(int)
    # values, counts = np.unique(labels_gm, return_counts=True)

    counts = counts.astype(float)
    counts /= np.sum(counts)
    log_Nm = np.log(counts)

    distance = np.load(original_directory+'diff_%s_%s_batch0.npy'%(dataset,filelabel))
    for i_batch in range(1,10):
        data = np.load(original_directory+'diff_%s_%s_batch%d.npy'%(dataset,filelabel,i_batch))
        distance = np.hstack((distance,data))
    
    M = distance.shape[0]
    N = distance.shape[1]

    norm = .5/(sigma**2)
    Dmat = -norm*distance.T

    ### Write to json format for Stan input

    dictionary = {
        "M": M,
        "N": N,
        "logNm": list(log_Nm),
        "Dmat": [list(a) for a in Dmat]
    }

    json_object = json.dumps(dictionary, indent=4)
    with open(filename, "w") as f:
        f.write(json_object)
        f.close()
    
    ### Run Stan for MCMC

    data_file = os.path.join('.', filename)
    fit = my_model.sample(data=data_file,
        chains=8,
        sig_figs=12,
        parallel_chains=8,
        threads_per_chain=20,
        iter_warmup=1000,
        iter_sampling=10000,
    )

    ### Save Stan output in csv format

    fit.save_csvfiles(dir=directory)

