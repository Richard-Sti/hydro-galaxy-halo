"""Matches galaxies to dark matter halos in H-AGN."""
import numpy as np
from astropy.cosmology import WMAP7

from scipy.optimize import minimize_scalar
from tqdm import tqdm


# Cosmology used for H-AGN
cosmo = WMAP7
nthreads = 2
boxsize = 100 / (cosmo.H0.to_value() / 100)

# Load the initial files
halos0 = np.loadtxt("../data/list_halo_00761.dat")
halos0_fits = np.genfromtxt("../data/halo_fit_particles.csv")
gals0 = np.loadtxt("../data/list_reffgal_00761.txt")

# Initialise the numpy arrays
halo_labels = ['x', 'y', 'z', 'Mvir', 'rs', 'rho0', 'concentration']
halos = np.zeros(
        halos0.shape[0], dtype={'names': halo_labels,
                                'formats': ['float64'] * len(halo_labels)})

gal_labels = ['x', 'y', 'z', 'MS', 'Reff']
gals = np.zeros(gals0.shape[0],
                dtype={'names': gal_labels,
                       'formats': ['float64'] * len(gal_labels)})
# Put the values into the new arrays and convert box coords. to Mpc
for i, coord in enumerate(['x', 'y', 'z']):
    i += 3
    halos[coord] = halos0[:, i] * boxsize
    gals[coord] = gals0[:, i] * boxsize

# Match fits back to catalog based on halo IDs and populate the arrays
IDs_catalog = halos0[:, 0].astype(int)
IDs_fits = halos0_fits[:, 0].astype(int)

for i, mask in tqdm(enumerate(IDs_catalog)):
    j = np.where(mask == IDs_fits)
    halos['rs'][i] = halos0_fits[j, 1]
    halos['rho0'][i] = halos0_fits[j, 2]

halos['Mvir'] = halos0[:, 2]
gals['MS'] = gals0[:, 2]
gals['Reff'] = gals0[:, 6] * boxsize


print('Calculating concentration')
X = halos['Mvir'] / (4 * np.pi * halos['rho0'] * halos['rs']**3)


def find_concentration(c, y):
    return abs(y - np.log(1 + c) + c / (1 + c))


bounds = [0, 10000]
for i in tqdm(range(X.size)):
    res = minimize_scalar(find_concentration, bounds=bounds, args=(X[i]),
                          method='bounded')
    if res['message'] != 'Solution found.':
        raise ValueError(res['message'])
    halos['concentration'][i] = res['x']


print('Saving')
np.save("../data/halos_00761.npy", halos)
np.save("../data/gals_00761761.npy", gals)
