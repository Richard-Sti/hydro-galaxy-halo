"""Matches galaxies to dark matter halos in H-AGN."""
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar
from tqdm import tqdm

NTHREADS = 12

ext_halos = np.loadtxt('../data/extracted_halos.txt')
fit_halos = np.genfromtxt('../data/halo_fit_particles.csv')
ext_gals = np.loadtxt('../data/extracted_gals.txt')

# Always make sure that these two match the extracted files
hcols = ['ID', 'level', 'parentID', 'x', 'y', 'z', 'r', 'rvir', 'Mvir',
         'rho0', 'spin', 'L', 'cvel']
gcols = ['ID', 'x', 'y', 'z', 'MS', 'spin']


def fit_concentration(c, Y):
    return abs(Y - np.log(1 + c) + c / (1 + c))


rho0 = fit_halos[:, 2]
rs = fit_halos[:, 1]
Mvir = ext_halos[:, hcols.index('Mvir')]
X = Mvir / (4 * np.pi * rho0 * rs**3)
concentration = np.zeros_like(Mvir)

bounds = [0, 10000]
for i in tqdm(range(X.size)):
    res = minimize_scalar(fit_concentration, bounds=bounds, args=(X[i]),
                          method='bounded')
    if res['message'] != 'Solution found.':
        raise ValueError(res['message'])
    concentration[i] = res['x']

names = ['ID', 'level', 'parentID', 'x', 'y', 'z', 'r', 'rho0', 'rvir', 'rs',
         'Mvir', 'spin', 'L', 'cvel', 'concentration']

halos = np.zeros(ext_halos.shape[0],
                 dtype={'names': names, 'formats': ['float64'] * len(names)})
gals = np.zeros(ext_gals.shape[0],
                dtype={'names': gcols, 'formats': ['float64'] * len(gcols)})


halos['ID'] = ext_halos[:, hcols.index('ID')]
halos['level'] = ext_halos[:, hcols.index('level')]
halos['parentID'] = ext_halos[:, hcols.index('parentID')]
halos['x'] = ext_halos[:, hcols.index('x')]
halos['y'] = ext_halos[:, hcols.index('y')]
halos['z'] = ext_halos[:, hcols.index('z')]
halos['r'] = ext_halos[:, hcols.index('r')]
halos['rho0'] = rho0
halos['rs'] = rs * 1e-3  # Mpc
halos['Mvir'] = ext_halos[:, hcols.index('Mvir')]
halos['spin'] = ext_halos[:, hcols.index('spin')]
halos['L'] = ext_halos[:, hcols.index('L')]
halos['cvel'] = ext_halos[:, hcols.index('cvel')]
halos['rvir'] = rs * concentration * 1e-3  # Mpc
halos['concentration'] = concentration


for i, p in enumerate(gcols):
    gals[p] = ext_gals[:, i]

print("Saving extracted halos and galaxies")
np.save('../data/extracted_halos.npy', halos)
np.save('../data/extracted_gals.npy', gals)


def find_match(i):
    """
    Finds the most massive galaxy within 5% of a halo's Rvir.

    The returned index corresponds to ordering in the galaxy array.
    """
    dx = np.array([gals[p] - halos[p][i] for p in ('x', 'y', 'z')]).T
    separation = np.linalg.norm(dx, axis=-1)

    mask = separation < 0.05 * halos['rvir'][i]
    if np.alltrue(np.logical_not(mask)):
        return np.nan

    IDS = np.where(mask)[0]
    return IDS[np.argmax(gals['MS'][IDS])]


print("Matching galaxies to halos.")
out = Parallel(n_jobs=NTHREADS)(delayed(find_match)(i)
                                for i in range(halos.size))


names = ['x', 'y', 'z', 'level', 'r', 'rvir', 'Mvir', 'rho0', 'spin', 'L',
         'cvel', 'MS', 'concentration']

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
    catalog['level'][k] = halos['level'][i]
    catalog['r'][k] = halos['r'][i]
    catalog['rvir'][k] = halos['rvir'][i]
    catalog['Mvir'][k] = halos['Mvir'][i]
    catalog['rho0'][k] = halos['rho0'][i]
    catalog['spin'][k] = halos['spin'][i]
    catalog['L'][k] = halos['L'][i]
    catalog['cvel'][k] = halos['cvel'][i]
    catalog['concentration'][k] = halos['concentration'][i]

    catalog['MS'][k] = gals['MS'][j]
    k += 1

np.save('../data/matched.npy', catalog)
