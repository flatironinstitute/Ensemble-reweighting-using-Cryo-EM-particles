import os
from cmdstanpy import CmdStanModel
import importlib.resources
import numpy as np
import json
import argparse
from typing import Dict, Any


def BuildCmdStanModel(stan_model_path: str = None, cpp_options: Dict[str, Any] = None):
    """
    Builds the Command Stan model from the script located at the given
    stan_model_path.

    Args:
        stan_model_path (str): Path to the stan model file.
        cpp_options (dict): Dictionary of options to compile the model with.

    Returns:
        CmdStanModel: CmdStan model object.
    """
    if cpp_options is None:
        cpp_options = {
            "STAN_THREADS": True,
        }
    if stan_model_path is None:
        module_path = importlib.resources.files("cryoER")
        path = os.path.join(module_path, "cryo-er.stan")

    my_model = CmdStanModel(
        stan_file=path,
        cpp_options=cpp_options,
    )
    return my_model


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
        "-l",
        "--lmbd",
        type=float,
        default=1e-6,
        help="standard deviation of colorless pixel noise in cryo-EM images",
    )
    parser.add_argument(
        "-fc",
        "--infileclustersize",
        default="cluster_size.txt",
        help="file containing the number of conformations in each cluster",
    )
    parser.add_argument(
        "-fd",
        "--infileimagedistance",
        nargs="+",
        default=[],
        help="(list of) file(s) containing the matrix of size M times N of pairwise distance between M structures and N images",
    )
    parser.add_argument(
        "-o", "--outdir", default="./output/", help="directory for output files"
    )
    parser.add_argument(
        "-nc",
        "--chains",
        type=int,
        default=1,
        help="number of MCMC chains for posterior sampling",
    )
    parser.add_argument(
        "-sf",
        "--sigfig",
        type=int,
        default=6,
        help="significant figures for MCMC outputs",
    )
    parser.add_argument(
        "-pc",
        "--parallelchain",
        type=int,
        default=1,
        help="(for parallelization) number of chains in parallel for MCMC",
    )
    parser.add_argument(
        "-tc",
        "--threadsperchain",
        type=int,
        default=1,
        help="(for parallelization) number of threads per chain for MCMC",
    )
    parser.add_argument(
        "-iw",
        "--iterwarmup",
        type=int,
        default=200,
        help="number of MCMC warm-up steps",
    )
    parser.add_argument(
        "-is",
        "--itersample",
        type=int,
        default=2000,
        help="number of MCMC warm-up steps",
    )

    return parser


def main(args=None):
    ######## Input parameters ########

    ######## ######## ######## ########
    if args is None:
        parser = _parse_args()
        args = parser.parse_args()

    lmbd = args.lmbd
    chains = args.chains
    sig_figs = args.sigfig
    parallel_chains = args.parallelchain
    threads_per_chain = args.threadsperchain
    iter_warmup = args.iterwarmup
    iter_sampling = args.itersample

    ## Read N_m, the number of conformations that are in the mth cluster
    infileclustersize = args.infileclustersize
    counts = np.loadtxt(infileclustersize).astype(int)
    counts = counts.astype(float)
    counts /= np.sum(counts)
    log_Nm = np.log(counts)

    ## Read distance matrix between cryoEM images and MD structures
    distance = None
    for f in args.infileimagedistance:
        if distance is None:
            distance = np.load(f)
        else:
            data = np.load(f)
            distance = np.hstack((distance, data))

    ## Write json files for reading into Stan program
    M = distance.shape[0]
    N = distance.shape[1]
    print("Number of structures = %d, Number of images = %d." % (M, N))

    outdir = args.outdir
    try:
        os.mkdir(outdir)
    except FileExistsError:
        pass

    ######## ######## ######## ########

    ## Compile Stan script
    my_model = BuildCmdStanModel()
    print(my_model.exe_info())

    json_filename = "%s/Dmat.json" % (outdir)
    stan_output_file = "%s/Stan_output" % (outdir)

    norm = 0.5 / (lmbd**2)
    Dmat = -norm * distance.T

    dictionary = {
        "M": M,
        "N": N,
        "logNm": list(log_Nm),
        "Dmat": [list(a) for a in Dmat],
    }

    json_object = json.dumps(dictionary, indent=4)

    with open(json_filename, "w") as f:
        f.write(json_object)
        f.close()

    ## Run Stan model to perform MCMC sampling on posterior in Eq. 10 and 17
    data_file = os.path.join(".", json_filename)
    fit = my_model.sample(
        data=data_file,
        chains=chains,
        sig_figs=sig_figs,
        parallel_chains=parallel_chains,
        threads_per_chain=threads_per_chain,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        show_console=True,
    )

    # Save Stan output, i.e., posterior samples, in CSV format, in a specified folder
    fit.save_csvfiles(dir=stan_output_file)


if __name__ == "__main__":
    main()
