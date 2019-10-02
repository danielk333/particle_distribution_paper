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

    samples = 15000
    samples_big = 100000

    experiment_name = 'analytic_is'
    experiment_ref = 'analytic_is'
    version_mcmc = 1
    version = 3
    version_big = 2

    fout_mcmc = f'./data/{experiment_name}_v{version_mcmc}_mcmc.pickle'
    fout_dmc = f'./data/{experiment_ref}_v{version}_dmc.pickle'
    fout_big_dmc1 = f'./data/{experiment_ref}_v{version_big}_big_dmc.pickle'
    fout_big_dmc2 = f'./data/{experiment_ref}_v{version_big}_big_dmc2.pickle'
    fout_is = f'./data/{experiment_name}_v{version_mcmc}_is.pickle'
    plot_folder = './plots_analytic/'

    target_dist = lambda vx, vy: \
        st.norm.pdf(vx,
                loc=0,
                scale=400,
            )\
        *st.norm.pdf(vy, 
                loc=0,
                scale=400,
            )
    
    source_cov = np.array(
    [[  5279.30222962, -36872.43373695],
     [-36872.43373695, 327762.2046178 ],
    ])
    source_mu = np.array([  96.79555751, -224.89482706])

    '''
    source_cov = np.array([
        [  2093.93165232, -12324.15647722],
        [-12324.15647722,  97890.13378184],
    ])
    source_mu = np.array([100.07146322, -176.42918555])
    '''
    def target_rng(num):
        model_state = np.random.randn(2, num)
        model_state[0,:] = model_state[0,:]*400 + 0
        model_state[1,:] = model_state[1,:]*400 + 0
        return model_state

    def source_rng(num):
        model_state = st.multivariate_normal.rvs(size=num, mean=source_mu, cov=source_cov).T
        return model_state


    def source_dist(vx, vy):
        return st.multivariate_normal.pdf(np.array([vx,vy]), mean=source_mu, cov=source_cov)



    results_source, chain_source = sampler.ReboundMinDist_DMC(
        t_end = t_end,
        rebound_opts = rebound_opts,
        rng = source_rng,
        samples = samples,
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
            method = 'DMC',
        )


        plt.show()

        exit()
