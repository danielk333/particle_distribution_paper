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


#Local
from . import functions
from .functions import AU
from .propagator import ReboundMinDist
from .decorators import pickle_cache

_year = 3600*24*365.25


def meteoroid_perihelion_model():

    init_orb = np.array([0.95*AU/(1.0 - 0.6), 0.6, 78, 180.0, 0.0, 180.0])
    init_cart = functions.kep2cart(init_orb, m=0.0, M_cent=functions.M_sol, radians=False)
    
    fw = init_cart[3:]/np.linalg.norm(init_cart[3:])
    side = init_cart[:3]/np.linalg.norm(init_cart[:3])

    def model(vx, vy):
        
        cart = init_cart.copy()
        
        pert = vx*fw + vy*side
        
        cart[3:] = cart[3:] + pert
        return cart


    return model


@pickle_cache
def ReboundMinDist_MCMC(
            t_end,
            rebound_opts,
            start,
            steps,
            chains,
            init_step,
            loglike,
            logprior = None,
        ):
    
    params = 2

    prop = ReboundMinDist(**rebound_opts)
    

    if logprior is None:
        logprior = lambda x: 0.0

    def logpost(y, x):
        return loglike(y) + logprior(x)

    model = meteoroid_perihelion_model()

    prop.states = np.empty((6,chains), dtype=np.float64)
    
    model_state = start.copy()
    for p in range(chains):
        prop.states[:, p] = model(*model_state[:, p].tolist())
    

    prop.t0 = 0.0

    step = np.array([init_step], np.float64)*10
    step = np.repeat(step.T, prop.num, axis = 1)
    
    t = np.arange(0, t_end*_year, prop.time_step, dtype=np.float64)

    chain = np.zeros((steps, params, prop.num), dtype=np.float64)
    #chain_prop = np.zeros((steps, 6, prop.num), dtype=np.float64)
    out_results = [[None]*prop.num for ind in range(steps)]
    accept = np.zeros((prop.num, 6), dtype=np.float64)
    tries = np.zeros((prop.num, 6), dtype=np.float64)

    logpost_now = np.zeros((prop.num, ), dtype=np.float64)
    logpost_try = np.zeros((prop.num, ), dtype=np.float64)

    results = prop.propagate(t)


    for p in range(prop.num):
        logpost_now[p] = logpost(results[p]['MinDist'], model_state[:, p])
    
    for ind in tqdm(range(steps)):

        x_current = np.copy(model_state)
        prop_current = np.copy(prop.states)
        
        for p in range(prop.num):
            pi = int(np.floor(np.random.rand(1)*params))

            d_step = np.random.randn(1)*step[pi, p]
            
            model_state[pi, p] += d_step

            prop.states[:, p] = model(*model_state[:,p].tolist())
        
        #save_state = prop.states.copy()
        save_model_state = model_state.copy()
        results = prop.propagate(t)

        for p in range(prop.num):
            logpost_try[p] = logpost(results[p]['MinDist'], model_state[:, p])
            
            alpha = np.log(np.random.rand(1))
            
            if logpost_try[p] > logpost_now[p]:
                _accept = True
            elif (logpost_try[p] - alpha) > logpost_now[p]:
                _accept = True
            else:
                _accept = False
            
            tries[p, pi] += 1.0
            
            if _accept:
                logpost_now[p] = logpost_try[p]
                accept[p, pi] += 1.0
            else:
                model_state[:,p] = x_current[:,p]
                prop.states[:, p] = prop_current[:, p]
            
            
            #print('log post {:<5.2e} AU'.format(logpost_try[p]/AU))

            if ind % 100 == 0 and ind > 0:
                for dim in range(params):
                    ratio = accept[p, dim]/tries[p, dim]

                    #print('ratio {:<5.2f}'.format(ratio))
                    if ratio > 0.5:
                        step[dim, p] *= 2.0
                    elif ratio < 0.3:
                        step[dim, p] /= 2.0
                    
                    accept[p, dim] = 0.0
                    tries[p, dim] = 0.0
            
            chain[ind, :, :] = save_model_state
            #chain_prop[ind, :, :] = save_state
            out_results[ind][p] = results[p]

    return out_results, chain

@pickle_cache
def ReboundMinDist_DMC(
            t_end,
            rebound_opts,
            rng,
            samples,
            parts = 1,
        ):

    params = 2
    prop = ReboundMinDist(**rebound_opts)

    model = meteoroid_perihelion_model()
    t = np.arange(0, t_end*_year, prop.time_step, dtype=np.float64)

    chain = np.zeros((1, params, samples), dtype=np.float64)
    model_state = rng(samples)

    for p in range(samples):
        chain[0, 0, p] = model_state[0,p]
        chain[0, 1, p] = model_state[1,p]

    assert samples//parts == samples/parts, 'parts need to divide samples'

    sub_samples = samples//parts
    results = []

    for ind in range(parts):

        prop.states = np.empty((6,sub_samples), dtype=np.float64)

        for p in range(sub_samples):
            prop.states[:, p] = model(*model_state[:,sub_samples*ind + p].tolist())
        
        prop.t0 = 0.0
        
        results += prop.propagate(t)

        prop = ReboundMinDist(**rebound_opts)

    return [results], chain



def combine_results(results_list, chain_list):

    results_comb = None
    chain_comb = None
    for results, chain in zip(results_list, chain_list):
        if results_comb is None or chain_comb is None:
            results_comb = results
            chain_comb = chain
        else:
            for ind in range(len(results)):
                results_comb[ind] += results[ind]
            chain_comb = np.append(chain_comb, chain, axis=2)
    return results_comb, chain_comb

