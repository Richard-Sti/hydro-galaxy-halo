# Copyright (C) 2021 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Matches galaxies to dark matter halos in H-AGN."""


import numpy as np
from joblib import Parallel, delayed


NTHREADS = 7

# Extracted halo properties
fit_genNFW = np.genfromtxt("/mnt/extraspace/deaglan/H-AGN/HALO_00761"
                           "/gennfw_fits/halo_fit_particles.csv")
fit_NFW = np.genfromtxt("/mnt/extraspace/deaglan/H-AGN/HALO_00761/"
                        "nfw_fits/halo_fit_particles.csv")
potential_grid = np.load("/mnt/extraspace/deaglan/Noise_Model/H-AGN/"
                         "761/grav_potential_761_N_512_kstar_None.npy")
particles = np.genfromtxt("/users/deaglan/Galaxy_Warps/"
                          "halo_nparticles_761.txt")
ext_halos = np.loadtxt('../data/extracted_halos.txt')
ext_gals = np.loadtxt('../data/extracted_gals.txt')
# Galaxy effective radius
Reff_gals = np.loadtxt('../data/list_reffgal_00761.txt')[:, 6]

print('Finished loading.')


# Always make sure that these two match the extracted files
hcols = ['ID', 'level', 'parentID', 'x', 'y', 'z', 'r', 'rvir', 'Mvir',
         'rho0', 'spin', 'L', 'cvel', 'Ekin', 'Epot', 'Eint']
gcols = ['ID', 'x', 'y', 'z', 'MS', 'spin', 'Reff']


# BIC to select better fit
BIC_NFW = 2 * np.log(particles[:, 1]) + 2 * fit_NFW[:, -1]
BIC_genNFW = 3 * np.log(particles[:, 1]) + 2 * fit_genNFW[:, -1]
dBIC = BIC_NFW = BIC_genNFW

gamma = fit_genNFW[:, 1]
rs = fit_genNFW[:, 2]
rho0 = fit_genNFW[:, 3]
Mvir = ext_halos[:, hcols.index('Mvir')]

# Where NFW fit has lower BIC by at least 2 replace GenNFW with NFW
mask_BIC = dBIC < -1.5
gamma[mask_BIC] = -1.
rs[mask_BIC] = fit_NFW[:, 1][mask_BIC]
rho0[mask_BIC] = fit_NFW[:, 2][mask_BIC]

# Where GenNFW did not converge replace with NFW
mask_convergence = fit_genNFW[:, -2] > -10
gamma[mask_convergence] == -1.
rs[mask_convergence] = fit_NFW[:, 1][mask_convergence]
rho0[mask_convergence] = fit_NFW[:, 2][mask_convergence]


names = ['ID', 'level', 'parentID', 'x', 'y', 'z', 'rho0', 'rvir', 'rs',
         'Mvir', 'spin', 'L', 'cvel', 'concentration', 'potential',
         'Ekin', 'Epot', 'Eint', 'gamma']

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
halos['rho0'] = rho0
halos['gamma'] = gamma
halos['rs'] = rs * 1e-3  # Mpc
halos['Mvir'] = Mvir
halos['spin'] = ext_halos[:, hcols.index('spin')]
halos['L'] = ext_halos[:, hcols.index('L')]
halos['cvel'] = ext_halos[:, hcols.index('cvel')]
halos['rvir'] = ext_halos[:, hcols.index('rvir')]  # Mpc
halos['concentration'] = halos['rvir'] / halos['rs']
halos['Ekin'] = ext_halos[:, hcols.index('Ekin')]
halos['Epot'] = -ext_halos[:, hcols.index('Epot')]
# Eint has funny values so apply this transformation
Eint = ext_halos[:, hcols.index('Eint')]
halos['Eint'] = np.sign(Eint) * np.log(np.abs(Eint) + 1)

# Extract the potential
N = potential_grid.shape[0]
xbins = np.digitize(halos['x'],
                    np.linspace(halos['x'].min(), halos['x'].max(), N)) - 1
ybins = np.digitize(halos['y'],
                    np.linspace(halos['y'].min(), halos['y'].max(), N)) - 1
zbins = np.digitize(halos['z'],
                    np.linspace(halos['z'].min(), halos['z'].max(), N)) - 1

halos['potential'] = potential_grid[xbins, ybins, zbins]

#  Populate galaxy array
for i, p in enumerate(gcols[:-1]):
    gals[p] = ext_gals[:, i]
gals['Reff'] = Reff_gals

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
                                for i in range(halos['Mvir'].size))

names = ['x', 'y', 'z', 'level', 'rvir', 'Mvir', 'rho0', 'gamma', 'spin', 'L',
         'cvel', 'MS', 'concentration', 'potential', 'Reff', 'parent_Mvir',
         'parent_L', 'parent_rho0', 'Ekin', 'Epot', 'parent_gamma', 'Eint',
         'parent_rvir', 'rs']

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
    catalog['rvir'][k] = halos['rvir'][i]
    catalog['Mvir'][k] = halos['Mvir'][i]
    catalog['rho0'][k] = halos['rho0'][i]
    catalog['spin'][k] = halos['spin'][i]
    catalog['rs'][k] = halos['rs'][i]
    catalog['gamma'][k] = halos['gamma'][i]
    catalog['Ekin'][k] = halos['Ekin'][i]
    catalog['Epot'][k] = halos['Epot'][i]
    catalog['Eint'][k] = halos['Eint'][i]
    catalog['L'][k] = halos['L'][i]
    catalog['cvel'][k] = halos['cvel'][i]
    catalog['concentration'][k] = halos['concentration'][i]
    catalog['potential'][k] = halos['potential'][i]
    catalog['MS'][k] = gals['MS'][j]
    catalog['Reff'][k] = gals['Reff'][j]

    parentID = int(halos['parentID'][i]) - 1
    catalog['parent_Mvir'][k] = halos['Mvir'][parentID]
    catalog['parent_L'][k] = halos['L'][parentID]
    catalog['parent_rho0'][k] = halos['rho0'][parentID]
    catalog['parent_gamma'][k] = halos['gamma'][parentID]
    catalog['parent_rvir'][k] = halos['rvir'][parentID]

    k += 1

np.save('../data/matched.npy', catalog)
