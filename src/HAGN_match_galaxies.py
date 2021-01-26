"""Matches galaxies to dark matter halos in H-AGN."""
import numpy as np

from joblib import Parallel, delayed


# import warnings
NTHREADS = 28

halos = np.load("../data/halos_00761.npy")
gals = np.load("../data/gals_00761761.npy")
Rvir = halos['rs'] * halos['concentration'] * 1e-3  # in Mpc


def find_match(i):
    """
    Finds the most massive galaxy within 10% of a halo's Rvir.

    The returned index corresponds to ordering in the galaxy array.
    """
    dx = np.array([gals[p] - halos[p][i] for p in ('x', 'y', 'z')]).T
    separation = np.linalg.norm(dx, axis=-1)

    mask = separation < 0.1 * Rvir[i]
    if np.alltrue(np.logical_not(mask)):
        return np.nan

    IDS = np.where(mask)[0]
    return IDS[np.argmax(gals['MS'][IDS])]


def find_potential(i):
    """Finds the potential due to the 20 closest halos."""
    dx = np.array([halos[p] - halos[p][i] for p in ('x', 'y', 'z')]).T
    separation = np.linalg.norm(dx, axis=-1)

    mask = np.argsort(separation)[1:21]
    potential = halos['Mvir'][mask] / separation[mask]
    return np.sum(potential)
#    mask = separation < 10 * halos[i]['Rvir']
#    with warnings.catch_warnings():
#        msg = "divide by zero encountered in true_divide"
#        warnings.filterwarnings("ignore", message=msg)
#        potential = halos['Mvir'][mask] / separation[mask]
#    if potential.size == 1:
#        return 0.
#    return np.sum(potential[np.isfinite(potential)])


print("Matching galaxies to halos.")
out = Parallel(n_jobs=NTHREADS)(delayed(find_match)(i)
                                for i in range(halos.size))
print("Finding potential.")
potential = Parallel(n_jobs=NTHREADS)(delayed(find_potential)(i)
                                      for i in range(halos.size))


names = ['x', 'y', 'z', 'logMvir', 'rs', 'rho0', 'concentration', 'Reff',
         'logMS', 'log_potential']
N = np.isfinite(out).sum()
catalog = np.zeros(N, dtype={'names': names,
                             'formats': ['float64'] * len(names)})


k = 0
for i, j in enumerate(out):
    if np.isnan(j):
        continue
    j = int(j)
    catalog['x'][k] = halos['x'][i]
    catalog['y'][k] = halos['y'][i]
    catalog['z'][k] = halos['z'][i]
    catalog['log_potential'][k] = np.log10(potential[i])
    catalog['logMvir'][k] = np.log10(halos['Mvir'][i])
    catalog['rs'][k] = halos['rs'][i]
    catalog['concentration'][k] = halos['concentration'][i]
    catalog['rho0'][k] = halos['rho0'][i]
    catalog['Reff'][k] = gals['Reff'][j]
    catalog['logMS'][k] = np.log10(gals['MS'][j])
    k += 1

np.save('../data/HAGN_matched_catalog.npy', catalog)
