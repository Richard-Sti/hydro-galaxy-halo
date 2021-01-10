"""Matches galaxies to dark matter halos in H-AGN."""
import numpy as np

from joblib import Parallel, delayed
from astropy.cosmology import WMAP7

from definitions import Rvir_from_mvir

import warnings

# Cosmology used for H-AGN
COSMO = WMAP7
NTHREADS = 2

def parse_data(halos0, gals0):
    """
    Parses the initial text files and outputs arrays with named columns.

    H-AGN specific.
    """
    boxsize = 100 / (COSMO.H0.to_value() / 100)
    halo_labels = ['x', 'y', 'z', 'Mvir', 'Rvir']
    halos = np.zeros(halos0.shape[0], dtype={'names': halo_labels,
                                 'formats': ['float64'] * len(halo_labels)})
    gal_labels = ['x', 'y', 'z', 'MS', 'Reff']
    gals = np.zeros(gals0.shape[0], dtype={'names': gal_labels,
                                'formats': ['float64'] * len(gal_labels)})
    # Put the values into the new arrays and convert box coords. to Mpc
    for cord, i in zip(('x', 'y', 'z'), (3, 4, 5)):
        halos[cord] = halos0[:, i] * boxsize
        gals[cord] = gals0[:, i] * boxsize

    halos['Mvir'] = halos0[:, 2]
    gals['MS'] = gals0[:, 2]
    gals['Reff'] = gals0[:, 6] * boxsize
    halos['Rvir'] = Rvir_from_mvir(halos['Mvir'], COSMO)
    return halos, gals


# Specific paths 
halos, gals = parse_data(np.loadtxt("../data/list_halo_00761.dat"),
                         np.loadtxt("../data/list_reffgal_00761.txt"))
print('Saving')
np.save("../data/halos.npy", halos)
np.save("../data/gals.npy", gals)

def find_match(i, halos, gals):
    """
    Finds the most massive galaxy within 10% of a halo's Rvir.

    The returned index corresponds to ordering in the galaxy array.
    """
    dx = np.array([gals[p] - halos[p][i] for p in ('x', 'y', 'z')]).T
    separation = np.linalg.norm(dx, axis=-1)

    mask = separation < 0.1 * halos[i]['Rvir']
    if np.alltrue(np.logical_not(mask)):
        return np.nan

    IDS = np.where(mask)[0]
    return IDS[np.argmax(gals['MS'][IDS])]


def find_potential(i, halos):
    """Finds the potential due to halos within 10 times Rvir."""
    dx = np.array([halos[p] - halos[p][i] for p in ('x', 'y', 'z')]).T
    separation = np.linalg.norm(dx, axis=-1)

    mask = separation < 10 * halos[i]['Rvir']
    with warnings.catch_warnings():
        msg = "divide by zero encountered in true_divide"
        warnings.filterwarnings("ignore", message=msg)
        potential = halos['Mvir'][mask] / separation[mask]
    if potential.size == 1:
        return 0.
    return np.sum(potential[np.isfinite(potential)])


out = Parallel(n_jobs=NTHREADS)(delayed(find_match)(i, halos, gals)
                         for i in range(halos.size))
potential = Parallel(n_jobs=NTHREADS)(delayed(find_potential)(i, halos)
                         for i in range(halos.size))

names = ['x', 'y', 'z', 'logMvir', 'Rvir', 'Reff', 'logMS', 'potential']
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
    catalog['potential'][k] = potential[i]
    catalog['logMvir'][k] = np.log10(halos['Mvir'][i])
    catalog['Rvir'][k] = halos['Rvir'][i]
    catalog['Reff'][k] = gals['Reff'][j]
    catalog['logMS'][k] = np.log10(gals['MS'][j])
    k += 1

np.save('../data/HAGN_matched_catalog.npy', catalog)
