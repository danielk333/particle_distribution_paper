import os
import sys
import pickle 

#Third party
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

#Project
from ismetlib import sampler
from ismetlib import statistics
from ismetlib import plots
from ismetlib import functions

AU = functions.AU


if __name__ == '__main__':

    if len(sys.argv) > 1:
        cmd = sys.argv[1].strip().lower()
    else:
        cmd = None

    t_end = 10.0 #in years
    rebound_opts = dict(
        time_step = 3600.0*24.0,
    )
    chain_n = 1
    samples = 15000
    samples_big = 100000
    step = [10.0, 10.0]
    start = np.empty((2,chain_n))
    start[:,0] = [100,0]

    experiment_name = 'mcmc_is'
    experiment_ref = 'analytic_is'
    version_mcmc = 3
    version = 3
    version_big = 2

    fout_mcmc = f'./data/{experiment_name}_v{version_mcmc}_mcmc.pickle'
    fout_dmc = f'./data/{experiment_ref}_v{version}_dmc.pickle'
    fout_big_dmc1 = f'./data/{experiment_ref}_v{version_big}_big_dmc.pickle'
    fout_big_dmc2 = f'./data/{experiment_ref}_v{version_big}_big_dmc2.pickle'
    fout_is = f'./data/{experiment_name}_v{version_mcmc}_is.pickle'
    plot_folder = './plots'

    target_dist = lambda vx, vy: \
        st.norm.pdf(vx,
                loc=0,
                scale=400,
            )\
        *st.norm.pdf(vy, 
                loc=0,
                scale=400,
            )

    def target_rng(num):
        model_state = np.random.randn(2, num)
        model_state[0,:] = model_state[0,:]*400 + 0
        model_state[1,:] = model_state[1,:]*400 + 0
        return model_state

    like_sigma = 0.03*AU
    trunc_norm_const = like_sigma*(1 - st.norm.cdf(0, loc=0, scale=like_sigma))
    norm_const = 1.0/np.sqrt(2*np.pi)
    like_C = np.log(trunc_norm_const*norm_const)

    def loglike(d):
        return -0.5*(d/like_sigma)**2 + like_C

    def logprior(x):
        _p = st.norm.logpdf(
            x[0], 
            loc=0,
            scale=1000,
        )
        _p += st.norm.logpdf(
            x[1], 
            loc=0,
            scale=1000,
        )
        return _p



    results_source, chain_source = sampler.ReboundMinDist_MCMC(
            t_end = t_end,
            rebound_opts = rebound_opts,
            start = start,
            steps = samples//chain_n,
            chains = chain_n,
            init_step = step,
            loglike = loglike,
            logprior = logprior,
            path = fout_mcmc,
        )

    results_target, chain_target = sampler.ReboundMinDist_DMC(
        t_end = t_end,
        rebound_opts = rebound_opts,
        rng = target_rng,
        samples = samples,
        path = fout_dmc,
    )

    
    results_true1, chain_true1 = sampler.ReboundMinDist_DMC(
        t_end = t_end,
        rebound_opts = rebound_opts,
        rng = target_rng,
        samples = samples_big,
        path = fout_big_dmc1,
        parts = 10,
    )
    results_true2, chain_true2 = sampler.ReboundMinDist_DMC(
        t_end = t_end,
        rebound_opts = rebound_opts,
        rng = target_rng,
        samples = samples_big,
        path = fout_big_dmc2,
        parts = 10,
    )

    results_list = [
        results_true1,
        results_true2,
    ]
    chain_list = [
        chain_true1,
        chain_true2,
    ]

    results_true, chain_true = sampler.combine_results(results_list, chain_list)
    samples_big *= 2

    '''
    data = np.empty((0, chain_source.shape[1]), dtype=chain_source.dtype)
    for ind in range(chain_source.shape[2]):
        data = np.append(data, chain_source[:,:,ind], axis=0)
    data = data[data[:,1] > 0,:]
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    sct = ax.scatter(data[:,0], np.arcsin(-data[:,1]/180.0*np.pi), marker='.', alpha=0.25)
    ax.set_xlabel('Initial velocity perturbation $\Delta v$ [m/s]', fontsize=22)
    ax.set_ylabel('Ejection angle $\\theta$ [deg]', fontsize=22)
    ax.tick_params('both', labelsize=22-4)
    plt.show()
    exit()
    '''

    data = np.empty((0, chain_source.shape[1]), dtype=chain_source.dtype)
    for ind in range(chain_source.shape[2]):
        data = np.append(data, chain_source[:,:,ind], axis=0)
    cov_est = np.cov(data.T)
    mu_est = np.mean(data, axis=0)
    
    print(cov_est)
    print(mu_est)

    #cov = np.diag( 1.06*std_est*(data.shape[0])**(-1./5.)*2 )
    
    cov = np.array([[100, 0], [0, 100]])*0.8
    print(cov)
    kde = statistics.construct_kde(chain_source, cov)

    def source_dist(vx, vy):
        return kde(np.array([vx, vy]))

    if cmd == 'sim' or cmd is None:
        exit()

    if cmd == 'sets':

        plots.plot_input_output_correlation(results_source, chain_source, tmin = 8, tmax = 10, plot_folder=plot_folder, name='Source sampling')
        plots.plot_input_output_correlation(results_target, chain_target, tmin = 8, tmax = 10, plot_folder=plot_folder, name='Target sampling')
        #plt.show()


        data_source, bins = statistics.format_histogram_data(results_source, chain_source, tmin=8, tmax=10, num=9)
        data_target, bins = statistics.format_histogram_data(results_target, chain_target, tmin=8, tmax=10, bins = bins)

        plots.plot_data_sets(data_source, bins=bins, plot_folder=plot_folder, name='Source sampling')
        plots.plot_data_sets(data_target, bins=bins, plot_folder=plot_folder, name='Target sampling')
        plt.show()

        exit()


    if cmd == 'pdf':

        plots.plot_kde(kde, res=200, xlim=[-800, 800], ylim=[-2000,2000], plot_folder=plot_folder, name='MCMC IC')

        plt.show()

        exit()


    if cmd == 'plot':

        #plot out-parameters and input params

        plots.plot_results(results_target, plot_folder, 'Target sampling', bins=100)
        plots.plot_results(results_source, plot_folder, 'Source sampling', bins=100)

        plots.plot_metric_and_chains(
            results_target, 
            chain_target, 
            plot_folder, 
            'Target sampling', 
        )
        plots.plot_metric_and_chains(
            results_source, 
            chain_source, 
            plot_folder, 
            'Source sampling',
        )

        plt.show()

        exit()



    if cmd == 'hist':


        data_source, bins = statistics.format_histogram_data(results_source, chain_source, tmin=8, tmax=10, num=9)
        data_target, bins = statistics.format_histogram_data(results_target, chain_target, tmin=8, tmax=10, bins = bins)
        data_true, bins = statistics.format_histogram_data(results_true, chain_true, tmin=8, tmax=10, bins = bins)


        true_hist, true_hist_err = statistics.direct_estimation(data_true, samples_big)
        hist, hist_err = statistics.direct_estimation(data_target, samples)

        _hists = statistics.importance_sampling_estimation(
            data_sets = data_source,
            samples = samples, 
            source_dist = source_dist, 
            target_dist = target_dist,
            path = fout_is,
        )
        sampling_hist, sampling_hist_err, estimated_hist, estimated_hist_err = _hists


        plots.histogram_comparison(
            bins = bins, 
            sampling_hist = sampling_hist,
            sampling_hist_err = sampling_hist_err,
            estimated_hist = estimated_hist, 
            estimated_hist_err = estimated_hist_err, 
            target_hist = hist, 
            target_hist_err = hist_err, 
            plot_folder = plot_folder, 
            name = 'Sampling test', 
            true_hist = true_hist, 
            true_hist_err = true_hist_err,
        )


        plt.show()

        exit()
