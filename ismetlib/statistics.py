#!/usr/bin/env python


#Python standard
import os
import pickle

#Third party
import numpy as np
import scipy
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from .functions import AU
from .decorators import pickle_cache


def construct_kde(chain, cov):

    #this is empirical, but is fine since this is post-simulation analysis
    d = chain.shape[1]
    cov_d = np.diag(cov)
    cov_inv = np.linalg.inv(cov)
    cov_det = np.linalg.det(cov)
    kernel = lambda x: (2.0*np.pi)**(-d/2.0)*cov_det**(-0.5)*np.exp(-0.5*(x.T @ cov_inv @ x))

    data = np.empty((0, d), dtype=chain.dtype)
    for ind in range(chain.shape[2]):
        data = np.append(data, chain[:,:,ind], axis=0)
    data = data.T

    def kde(x):
        _p = 0.0
        
        inds = np.abs(data[0,:] - x[0]) < 3*cov_d[0]
        for ind in range(1,len(x)):
            inds = np.logical_or(inds, np.abs(data[ind,:] - x[ind]) < 3*cov_d[ind])

        data_t = data[:, inds]
        l = data_t.shape[1]
        if l > 0:
            for ind in range(l):
                _p += kernel(x - data_t[:,ind])
            _p /= float(l)

        return _p

    return kde


def format_histogram_data(results, chain, tmin, tmax, bins=None, num=10):

    #chain has format: iteration, parameter, chain index

    #collect data from chains and concatenate input with output data from simulations
    all_data = []
    for res, inps_ind in zip(results, range(chain.shape[0])):
        for out, inp_ind in zip(res, range(chain.shape[2])):
            all_data.append([out['MinDist'], out['t']] + chain[inps_ind, :, inp_ind].tolist())
            #dist, time, ejection vel, ejection angle
    all_data = np.array(all_data)

    #form the sets we are interested in by using the output space parameters
    filt_data = all_data[all_data[:,0] < 0.1*AU, :] #only inside hill sphere
    filt_data = filt_data[filt_data[:, 1] >= 3600*24*365.25*tmin, :]
    filt_data = filt_data[filt_data[:, 1] <= 3600*24*365.25*tmax, :]

    print(f'Data reduction {filt_data.shape[0]}/{all_data.shape[0]}')

    if bins is None:
        #we want distributions over time
        _, bins = np.histogram(filt_data[:,1], num)

    data_sets = []
    for ind in range(len(bins)-1):
        select = np.logical_and(filt_data[:, 1] > bins[ind], filt_data[:, 1] <= bins[ind+1])
        data_sets.append( filt_data[select, :] )
    
    return data_sets, bins, 


def direct_estimation(data_sets, samples):

    hist = np.empty((len(data_sets),))
    hist_err = np.empty((len(data_sets),))

    for ind in range(len(data_sets)):
        
        hist[ind] = float(data_sets[ind].shape[0])/float(samples)
        hist_err[ind] = np.sqrt((hist[ind]*(1 - hist[ind]))/float(samples))

    return hist, hist_err


@pickle_cache
def importance_sampling_estimation(data_sets, samples, source_dist, target_dist):

    sampling_hist = np.empty((len(data_sets),))
    estimated_hist = np.empty((len(data_sets),))

    sampling_hist_err = np.empty((len(data_sets),))
    estimated_hist_err = np.empty((len(data_sets),))

    pbar = tqdm(total=len(data_sets), ncols=60)

    for di, data in enumerate(data_sets):
        pbar.update(1)

        sampling_hist[di] = float(len(data))/float(samples)
        sampling_hist_err[di] = np.sqrt((sampling_hist[di]*(1- sampling_hist[di]))/float(samples))

        h_samples = np.empty((len(data),))
        source_data = np.empty((len(data),))
        target_data = np.empty((len(data),))
        for ind in range(len(data)):
            target_data[ind] = target_dist(data[ind,2], data[ind,3])
            source_data[ind] = source_dist(data[ind,2], data[ind,3])

        h_samples = target_data/source_data

        p_tmp = float(len(data))/float(samples)

        #calculate the monte-carlo integration
        estimated_hist[di] = np.mean(h_samples)*p_tmp
        estimated_hist_err[di] = np.std(h_samples)*p_tmp/np.sqrt(len(data))

    return sampling_hist, sampling_hist_err, estimated_hist, estimated_hist_err
