"""Matches galaxies to dark matter halos in H-AGN."""
import numpy as np

from joblib import Parallel, delayed


# import warnings
NTHREADS = 28


halos = np.load("/mnt/zfsusers/rstiskalek/hydro/data/halos_00761.npy")
gals = np.load("/mnt/zfsusers/rstiskalek/hydro/data/gals_00761761.npy")
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


print("Matching galaxies to halos.")
out = Parallel(n_jobs=NTHREADS)(delayed(find_match)(i)
                                for i in range(halos.size))

names = ['x', 'y', 'z', 'logMvir', 'rs', 'rho0', 'concentration', 'Reff',
         'logMS', 'Rvir']
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
    catalog['logMvir'][k] = np.log10(halos['Mvir'][i])
    catalog['rs'][k] = halos['rs'][i]
    catalog['concentration'][k] = halos['concentration'][i]
    catalog['rho0'][k] = halos['rho0'][i]
    catalog['Rvir'][k] = (halos['rs'] * halos['concentration'])[i]
    catalog['Reff'][k] = gals['Reff'][j] * 1000  # in kpc
    catalog['logMS'][k] = np.log10(gals['MS'][j])
    k += 1

# Eliminate halos with suspiciously high concentration
catalog = catalog[catalog['concentration'] < 300]

np.save('/mnt/zfsusers/rstiskalek/hydro/data/HAGN_matched_catalog.npy',
        catalog)
