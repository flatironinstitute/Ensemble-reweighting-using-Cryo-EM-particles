from cryoER import run_cryoER_mcmc as rcmc

def test_cmd_stan_builds():
    model = rcmc.BuildCmdStanModel()
    print(model)
