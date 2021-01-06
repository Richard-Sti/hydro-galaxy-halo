"""Matches galaxies to dark matter halos in H-AGN."""
import numpy as np

from joblib import Parallel, delayed

from astropy.cosmology import WMAP7
from astropy import units as u


def calculate_Rvir(mvir, cosmo, z=0):
    """Calculates the halo virial radius, assuming a flat universe."""
    # Calculate the halo virial radius
    mvir = mvir.copy() * u.Msun
    Omega = cosmo.Om0 * (1 + z)**3 / cosmo.efunc(z)**2
    x = Omega - 1
    # Overdensity parameter
    Delta_vir = 18 * np.pi**2 + 82 * x - 39 * x**2
    # Critical density today
    rho_crit = cosmo.critical_density(z).to(u.kg / u.m**3)
    # Virial radius
    Rvir = (3 / (4 * np.pi) * mvir.to(u.kg) / (rho_crit * Delta_vir))**(1/3)
    # Get in Mpc
    Rvir = Rvir.to(u.Mpc).to_value()
    return Rvir


# Cosmology used for H-AGN
cosmo = WMAP7

# Store the input data
halos0 = np.loadtxt("/mnt/extraspace/jeg/greenwhale/Sugata/H-AGN/Catalogs/"
                    "Halos/list_halo_00761.dat")
gals0 = np.loadtxt("/mnt/extraspace/jeg/greenwhale/Sugata/H-AGN/Catalogs/"
                   "Gals/list_reffgal_00761.txt")

boxsize = 100 / (cosmo.H0.to_value() / 100)
names_h = ['x', 'y', 'z', 'Mvir', 'Rvir']
N_h = halos0.shape[0]
halos = np.zeros(N_h, dtype={'names': names_h,
                             'formats': ['float64'] * len(names_h)})
names_g = ['x', 'y', 'z', 'MS', 'Reff']
N_g = gals0.shape[0]
gals = np.zeros(N_g, dtype={'names': names_g,
                            'formats': ['float64'] * len(names_g)})

for cord, i in zip(('x', 'y', 'z'), (3, 4, 5)):
    halos[cord] = halos0[:, i] * boxsize
    gals[cord] = gals0[:, i] * boxsize

halos['Mvir'] = halos0[:, 2]
gals['MS'] = gals0[:, 2]
gals['Reff'] = gals0[:, 6]

# Calculate the virial radius
halos['Rvir'] = calculate_Rvir(halos['Mvir'], cosmo)


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


out = Parallel(n_jobs=6)(delayed(find_match)(i, halos, gals)
                         for i in range(halos.size))

names = ['x', 'y', 'z', 'logMvir', 'Rvir', 'Reff', 'logMS']
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
    catalog['Rvir'][k] = halos['Rvir'][i]
    catalog['Reff'][k] = gals['Reff'][j]
    catalog['logMS'][k] = np.log10(gals['MS'][j])
    k += 1

np.save('/mnt/zfsusers/rstiskalek/hydro/data/HAGN_matched_catalog.npy',
        catalog)
