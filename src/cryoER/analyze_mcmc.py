import argparse
import os
import csv
import numpy as np
from scipy.special import logsumexp

def _parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(
        prog="cryoERMCMC",
        description="Perform MCMC sampling with Stan to sample posterior of weights ",
        # epilog = 'Text at the bottom of help'
    )
    parser.add_argument(
        "-o", "--outdir", default="./output/", help="directory for output files"
    )
    parser.add_argument(
        "-fc",
        "--infileclustersize",
        default="cluster_size.txt",
        help="file containing the number of conformations in each cluster",
    )
    return parser


def analyze_mcmc(
    output_directory = './output/',
    filename_cluster_counts = 'cluster_counts.txt',
):

    stan_directory = output_directory + 'Stan_output/'

    nm = np.loadtxt(filename_cluster_counts)
    N_center = len(nm)
    log_weights = []
    lp = []
    files = sorted(os.listdir(stan_directory))
    print(files)

    nm /= np.sum(nm)
    log_nm = np.log(nm)

    log_weights_d = []
    for file in files:
        log_weights_chain = []
        lp_chain = []
        with open('%s/%s'%(stan_directory, file), newline='') as csvfile:
            reader = csv.DictReader(filter(lambda row: row[0]!='#', csvfile), )
            for row in reader:
                log_weights_row = [float(row["log_weights.%d"%i]) for i in range(1,N_center+1)]
                lp_chain.append(float(row["lp__"]))
                log_weights_chain.append(log_weights_row)
        log_weights = np.array(log_weights_chain)
        lp_chain = np.array(lp_chain)
        log_weights_d.append(log_weights)
        lp.append(lp_chain)

    log_weights_d = np.array(log_weights_d)

    log_factor = log_weights_d - logsumexp(log_weights_d, axis=2)[:,:,None]

    factor = np.exp(log_factor)         # sampled reweighting factor
    factor_mean = factor.mean((0,1))
    factor_std = factor.std((0,1))

    factor_mean_std = np.vstack((factor_mean, factor_std)).T
    np.savetxt(output_directory+"reweighting_factor.txt", factor_mean_std, fmt='%.6f')

    log_rewtprob = log_weights_d + log_nm[None,None,:]
    log_rewtprob -= logsumexp(log_rewtprob, axis=2)[:,:,None]

    rewtprob = np.exp(log_rewtprob)     # reweighted probability
    rewtprob_mean = rewtprob.mean((0,1))
    rewtprob_std = rewtprob.std((0,1))

    rewtprob_mean_std = np.vstack((rewtprob_mean, rewtprob_std)).T
    np.savetxt(output_directory+"reweighted_prob.txt", rewtprob_mean_std, fmt='%.6f')

    lp = np.array(lp)
    np.savetxt(output_directory+"lp.txt", lp, fmt='%.6f')


if __name__ == "__main__":

    parser = _parse_args()
    args = parser.parse_args()
    output_directory = args.outdir
    filename_cluster_counts = args.infileclustersize

    analyze_mcmc(
        output_directory = output_directory,
        filename_cluster_counts = filename_cluster_counts,
    )

    