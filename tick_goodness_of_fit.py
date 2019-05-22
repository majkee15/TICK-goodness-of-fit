import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy import stats
from tick.hawkes import SimuHawkes, HawkesKernelTimeFunc
import itertools


def resid(x, intensities, timestamps, dim, method):
    arrivals = timestamps[dim]
    thetas = np.zeros(len(arrivals) - 1)
    ints = intensities[dim]
    for i in range(1, len(arrivals)):
        mask = (x <= arrivals[i]) & (x >= arrivals[i - 1])
        xs = x[mask]
        ys = ints[mask]
        try:
            thetas[i - 1] = method(ys, xs)
        except:
            thetas[i - 1] = np.nan

    return thetas


def goodness_of_fit_par(learner, arrivals, step, method):
    dimension = learner.n_nodes
    intensities = learner.estimated_intensity(arrivals, step)[0]
    x = learner.estimated_intensity(arrivals, step)[1]
    residuals = [resid(x, intensities, arrivals, dim, method) for dim in range(dimension)]
    return residuals


def plot_resid(resid, rows, cols):
    fig, axes = plt.subplots(nrows=rows, ncols=cols)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle('Goodness-of-fit for nonparametric HP')

    for ax, res in zip(axes, resid):
        k = stats.probplot(res, dist=stats.expon, fit=True, plot=ax, rvalue=False)
        ax.plot(k[0][0], k[0][0], 'k--')
        print(stats.kstest(res[np.logical_not(np.isnan(res))], 'expon'))


def goodness_of_fit_nonpar(nplearner, arrivals, timestep, method):
    # setup simulation object
    hp = SimuHawkes(baseline=nplearner.baseline)
    # set kernels
    support = np.arange(0, np.max(nplearner.get_kernel_supports()), timestep)

    for i, j in itertools.product(range(nplearner.n_nodes), repeat=2):
        print('Kernel {} set.'.format([i, j]))
        y = nplearner.get_kernel_values(i, j, support)
        kernel = HawkesKernelTimeFunc(t_values=support, y_values=y)
        hp.set_kernel(i, j, kernel)

    hp.track_intensity(timestep)
    hp.set_timestamps(timestamps=arrivals)
    dimension = hp.n_nodes
    intensities = hp.tracked_intensity
    x = hp.intensity_tracked_times
    residuals = [resid(x, intensities, arrivals, dim, method) for dim in range(dimension)]
    return residuals

def goodness_of_fit_nonpar_paralel(nplearner, arrivals, timestep, method):
    pool = mp.Pool(mp.cpu_count())
    # setup simulation object
    hp = SimuHawkes(baseline = nplearner.baseline)
    # set kernels
    support = np.arange(0,np.max(nplearner.get_kernel_supports()),timestep)
    
    for i, j in itertools.product(range(nplearner.n_nodes), repeat=2):
        print('Kernel {} set.'.format([i,j]))
        y = nplearner.get_kernel_values(i,j,support)
        kernel = HawkesKernelTimeFunc(t_values=support, y_values=y)
        hp.set_kernel(i, j, kernel)
        
    hp.track_intensity(timestep)
    hp.set_timestamps(timestamps = arrivals)    
    dimension = hp.n_nodes
    intensities = hp.tracked_intensity
    x = hp.intensity_tracked_times
    residuals = [pool.apply(resid,(x,intensities,arrivals, dim, method)) for dim in range(dimension)]
    pool.close()
    return residuals