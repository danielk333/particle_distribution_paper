
#Python standard
import os
import pickle

#Third party
import numpy as np
import scipy
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import matplotlib as mpl

#Local
from . import functions
from .functions import AU

#turn on TeX interperter
plt.rc('text', usetex=True)
#plt.style.use('dark_background')
#mpl.rc('image', cmap='cool')


_font = 22
_font_t = 28
_y = 3600.0*24.0*365.25


def plot_kde(kde, res, xlim, ylim, plot_folder, name):

    xv = np.linspace(xlim[0], xlim[1], num=res)
    yv = np.linspace(ylim[0], ylim[1], num=res)

    pi = np.empty((res,res))

    pbar = tqdm(total = res**2, ncols=60)

    for xi, x in enumerate(xv):
        for yi, y in enumerate(yv):
            pbar.update(1)
            pi[xi, yi] = kde(np.array([x, y]))

    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    hst = ax.pcolor(xv, yv, pi.T, cmap='viridis')

    ax.set_title('Full reconstruction of $\pi$ using KDE', fontsize=_font_t)
    ax.set_xlabel('Along orbit $v_x$ [m/s]', fontsize=_font)
    ax.set_ylabel('Across orbit $v_y$ [m/s]', fontsize=_font)
    ax.tick_params('both', labelsize=_font-4)

    cbar = plt.colorbar(hst, ax=ax)
    cbar.set_label(
        'Probability',
        fontsize=_font - 3,
    )

    fig.savefig(plot_folder + '/{}_KDE_PDF.png'.format(name.replace(' ', '_')),bbox_inches='tight')


def plot_data_sets(data_sets, bins, plot_folder, name):

    fig, ax = plt.subplots(1, 2, figsize=(12,8))

    data_c = cm.get_cmap('hsv', len(data_sets))

    for di, data in enumerate(data_sets):

        if len(data) > 0:

            ax[0].plot(data[:,2], data[:,3], '.', alpha=0.25, color=data_c(float(di)/float(len(data_sets))))
            ax[1].bar(bins[di]/_y, len(data), width=(bins[di+1]-bins[di])/_y, align = 'edge', facecolor=data_c(float(di)/float(len(data_sets))))

            ax[0].set_xlabel('Along orbit $v_x$ [m/s]', fontsize=_font)
            ax[0].set_ylabel('Across orbit $v_y$ [m/s]', fontsize=_font)
            ax[1].set_xlabel('Time [y]', fontsize=_font)
            ax[1].set_ylabel('Frequency', fontsize=_font)
            ax[0].tick_params('both', labelsize=_font-4)
            ax[1].tick_params('both', labelsize=_font-4)

    plt.suptitle('Integration regions in input vs output space', fontsize=_font_t)

    plt.tight_layout(w_pad=2)
    fig.savefig(plot_folder + '/{}_integration.png'.format(name.replace(' ', '_')),bbox_inches='tight')

def plot_input_output_correlation(results, chain, plot_folder, name, tmin = None, tmax = None):

    all_data = []
    for res, inps_ind in zip(results, range(chain.shape[0])):
        for out, inp_ind in zip(res, range(chain.shape[2])):
            all_data.append([out['MinDist'], out['t']] + chain[inps_ind, :, inp_ind].tolist())
    all_data = np.array(all_data)

    if tmin is not None:
        all_data = all_data[all_data[:, 1] >= _y*tmin, :]
    if tmax is not None:
        all_data = all_data[all_data[:, 1] <= _y*tmax, :]

    fig, axs = plt.subplots(1, 2, figsize=(12,8), sharex=True, sharey=True)
    ax = axs[0]
    sct = ax.scatter(all_data[:,2], all_data[:,3], c=all_data[:,1]/_y, marker='.', alpha=0.25)
    cbar = plt.colorbar(sct, ax=ax)
    cbar.set_label(
        'Encounter time [y]',
        fontsize=_font - 3,
    )
    cbar.ax.tick_params(labelsize=_font-4)

    ax.set_xlabel('Along orbit $v_x$ [m/s]', fontsize=_font)
    ax.set_ylabel('Across orbit $v_y$ [m/s]', fontsize=_font)
    ax.set_title('Input data vs encounter time', fontsize=_font_t)
    ax.tick_params('both', labelsize=_font-4)

    ax = axs[1]
    sct = ax.scatter(all_data[:,2], all_data[:,3], c=all_data[:,0]/AU, marker='.', alpha=0.25)
    cbar = plt.colorbar(sct, ax=ax)
    cbar.set_label(
        'Encounter distance [AU]',
        fontsize=_font - 3,
    )
    cbar.ax.tick_params(labelsize=_font-4)

    ax.set_xlabel('Along orbit $v_x$ [m/s]', fontsize=_font)
    ax.set_ylabel('Across orbit $v_y$ [m/s]', fontsize=_font)
    ax.set_title('Input data vs encounter distance', fontsize=_font_t)
    ax.tick_params('both', labelsize=_font-4)

    plt.tight_layout(w_pad=2)
    fig.savefig(plot_folder + '/{}_phi_map.png'.format(name.replace(' ', '_')),bbox_inches='tight')

def histogram_comparison(
        bins, 
        sampling_hist,
        sampling_hist_err,
        estimated_hist, 
        estimated_hist_err, 
        target_hist, 
        target_hist_err, 
        plot_folder, 
        name, 
        true_hist = None, 
        true_hist_err = None,
        method = 'MCMC',
    ):

    y_width = (bins[-2] - bins[-1])/_y

    fig, ax = plt.subplots(1, 1, figsize=(12,8))

    ax.bar(bins[:-1]/_y, target_hist, width=y_width, label='DMC sampling (3 $\\sigma$)', alpha=0.4, color='c', edgecolor = 'black', linewidth=2)
    ax.bar(bins[:-1]/_y, estimated_hist, width=y_width, label=f'{method} importance sampling (3 $\\sigma$)', alpha=0.5, color='r', edgecolor = 'black', linewidth=2)
    ax.errorbar(bins[:-1]/_y, target_hist, yerr=target_hist_err*3, fmt='.', alpha=0.6, color='c', capsize=20, elinewidth=3)
    ax.errorbar(bins[:-1]/_y, estimated_hist, yerr=estimated_hist_err*3, fmt='.', alpha=0.8, color='r', capsize=20, elinewidth=3)
    if true_hist is not None:
        ax.errorbar(bins[:-1]/_y, true_hist, yerr=true_hist_err*3, color='g', capsize=10, elinewidth=3, fmt='*-', label='Large DMC sampling (3 $\\sigma$)')
    ax.set_xlabel('Time [y]', fontsize=_font)
    ax.set_ylabel('Probability', fontsize=_font)
    ax.set_title(f'{method} importance sampling estimated probability', fontsize=_font_t)
    ax.yaxis.grid(True)
    ax.set_ylim(bottom=0)
    ax.tick_params('both', labelsize=_font-4)
    ax.legend(fontsize=_font, loc='lower left')

    fig.savefig(plot_folder + '/{}_{}_DMC_compare.png'.format(name.replace(' ', '_'), method),bbox_inches='tight')


    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    

    ax.bar(bins[:-1]/_y, target_hist_err, width=y_width, label='DMC sampling', alpha=0.4, color='c')
    ax.bar(bins[:-1]/_y, estimated_hist_err, width=y_width, label='MCMC importance sampling', alpha=0.5, color='r')
    ax.set_xlabel('Time [y]', fontsize=_font)
    ax.set_ylabel('Estimated Error', fontsize=_font)
    ax.set_title('MCMC importance sampling versus DMC error comparison', fontsize=_font_t)
    ax.legend(fontsize=_font)
    
    fig.savefig(plot_folder + '/{}_MCMCIC_DMC_error_compare.png'.format(name.replace(' ', '_')),bbox_inches='tight')



    fig, ax = plt.subplots(2, 1, figsize=(12,8))
    
    ax[0].bar(bins[:-1]/_y, target_hist, width=y_width, label='DMC sampling', color='c')
    ax[1].bar(bins[:-1]/_y, sampling_hist, width=y_width, label='MCMC sampling', color='c')
    ax[0].set_xlabel('Time [y]', fontsize=_font)
    ax[0].set_ylabel('Probability', fontsize=_font)
    ax[0].set_title('MCMC sampling versus DMC', fontsize=_font_t)
    ax[1].set_xlabel('Time [y]', fontsize=_font)
    ax[1].set_ylabel('Probability', fontsize=_font)
    ax[0].legend(fontsize=_font)
    ax[1].legend(fontsize=_font)

    fig.savefig(plot_folder + '/{}_MCMC_DMC_split.png'.format(name.replace(' ', '_')),bbox_inches='tight')


    fig, ax = plt.subplots(2, 1, figsize=(12,8))
    
    ax[0].bar(bins[:-1]/_y, target_hist, width=y_width, label='DMC sampling', color='c')
    ax[1].bar(bins[:-1]/_y, estimated_hist, width=y_width, label='MCMC importance sampling', color='c')
    ax[0].set_xlabel('Time [y]', fontsize=_font)
    ax[0].set_ylabel('Probability', fontsize=_font)
    ax[0].set_title('MCMC distribution renormalization by MC integration', fontsize=_font_t)
    ax[1].set_xlabel('Time [y]', fontsize=_font)
    ax[1].set_ylabel('Probability', fontsize=_font)
    ax[0].legend(fontsize=_font)
    ax[1].legend(fontsize=_font)

    fig.savefig(plot_folder + '/{}_MCMCIC_DMC_split.png'.format(name.replace(' ', '_')),bbox_inches='tight')

    fig, ax = plt.subplots(2, 1, figsize=(12,8))
    
    ax[0].bar(bins[:-1]/_y, sampling_hist, width=y_width, label='MCMC sampling', color='c')
    ax[1].bar(bins[:-1]/_y, estimated_hist, width=y_width, label='MCMC importance sampling', color='c')
    ax[0].set_title('MCMC distribution renormalization by MC integration', fontsize=_font_t)
    ax[0].set_xlabel('Time [y]', fontsize=_font)
    ax[0].set_ylabel('Probability', fontsize=_font)
    ax[1].set_xlabel('Time [y]', fontsize=_font)
    ax[1].set_ylabel('Probability', fontsize=_font)
    ax[0].legend(fontsize=_font)
    ax[1].legend(fontsize=_font)

    fig.savefig(plot_folder + '/{}_MCMC_MCMCIC_split.png'.format(name.replace(' ', '_')),bbox_inches='tight')



def plot_results(results, plot_folder, name, bins=None):

    fig, ax = plt.subplots(2, 1, figsize=(12,8))

    _dst = []
    for ch in results:
        for x in ch:
            _dst.append(x['MinDist']/AU)

    ax[0].hist(_dst, bins=bins)

    ax[0].set_title(f'{name} output distributions', fontsize=_font_t)
    ax[0].set_xlabel('Earth Distance [AU]', fontsize=_font)
    ax[0].set_ylabel('Frequency', fontsize=_font)
    ax[0].tick_params('both', labelsize=_font-4)

    _dst = []
    for ch in results:
        for x in ch:
            _dst.append(x['t']/(3600.0*24.0*365.25))

    ax[1].hist(_dst, bins=bins)

    ax[1].set_xlabel('Encounter time [y]', fontsize=_font)
    ax[1].set_ylabel('Frequency', fontsize=_font)
    ax[1].tick_params('both', labelsize=_font-4)

    plt.tight_layout(h_pad=2)
    fig.savefig(plot_folder + '/{}_output_hists.png'.format(name.replace(' ', '_')),bbox_inches='tight')
    


def plot_metric_and_chains(res, ch, plot_folder, name, limits = None):
    
    
    t_moid = []
    #points = []
    #points_E = []
    for p in range(ch.shape[2]):
        t_moid += [[]]
        #points += [np.empty((6,1), dtype=np.float64)]
        #points_E += [np.empty((6,1), dtype=np.float64)]
        for result in res:
            t_moid[p] += [result[p]['MinDist']]
            #points[p] = np.append(points[p], np.reshape(result[p]['state'], (6,1)), axis=1)
            #points_E[p] = np.append(points_E[p], np.reshape(result[p]['state_E'], (6,1)), axis=1)
            
        t_moid[p] = np.array(t_moid[p])
    
    '''
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12,8))
    for p in range(ch.shape[2]):
        ax.plot(
            (points[p][0,:] - points_E[p][0,:])/AU,
            (points[p][1,:] - points_E[p][1,:])/AU,
            '.',
        )

    xc = np.cos(np.linspace(0, np.pi*2, 100))*0.01
    yc = np.sin(np.linspace(0, np.pi*2, 100))*0.01

    ax.plot(xc, yc, '-r')

    ax.set_title(f'{name} particle closest encounter in Earth-fixed frame', fontsize=_font_t)
    ax.set_xlabel('X-ECI [AU]', fontsize=_font)
    ax.set_ylabel('Y-ECI [AU]', fontsize=_font)

    fig.savefig(plot_folder + '/{}_ECI_dist.png'.format(name.replace(' ', '_')),bbox_inches='tight')
    '''
    
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12,8))

    for p in range(ch.shape[2]):
        ax.plot(ch[:, 0, p], ch[:, 1, p], '.b', alpha=0.2)

    if limits is not None:
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])

    ax.set_title(f'{name} input distribution sampling', fontsize=_font_t)
    ax.set_xlabel('Along orbit $v_x$ [m/s]', fontsize=_font)
    ax.set_ylabel('Across orbit $v_y$ [m/s]', fontsize=_font)
    ax.tick_params('both', labelsize=_font-4)
    
    fig.savefig(plot_folder + '/{}_model_param_dist.png'.format(name.replace(' ', '_')),bbox_inches='tight')


    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12,8))

    for p in range(ch.shape[2]):
        axes[0].plot(ch[:, 0, p])
        axes[1].plot(ch[:, 1, p])
        axes[2].plot(t_moid[p]/AU)


    axes[0].set_title(f'{name} sampling chains', fontsize=_font_t)

    axes[0].set_ylabel('Along orbit $v_x$ [m/s]', fontsize=_font)
    axes[1].set_ylabel('Across orbit $v_y$ [m/s]', fontsize=_font)
    axes[0].tick_params('both', labelsize=_font-4)
    axes[1].tick_params('both', labelsize=_font-4)
    axes[2].set_xlabel('Sample number', fontsize=_font)
    axes[2].set_ylabel('$d$ [AU]', fontsize=_font)
    axes[2].tick_params('both', labelsize=_font-4)

    plt.tight_layout(h_pad=2)
    fig.savefig(plot_folder + '/{}_chains.png'.format(name.replace(' ', '_')),bbox_inches='tight')