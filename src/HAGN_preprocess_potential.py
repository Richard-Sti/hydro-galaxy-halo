"""Code in development to find the Newtonian potential due to nearby halos."""
import numpy as np

from joblib import Parallel, delayed

# import warnings
NTHREADS = 28


halos = np.load("/mnt/zfsusers/rstiskalek/hydro/data/halos_00761.npy")
gals = np.load("/mnt/zfsusers/rstiskalek/hydro/data/gals_00761761.npy")
Rvir = halos['rs'] * halos['concentration'] * 1e-3  # in Mpc

def find_potential(i):
    """Finds the potential due all halos within the 10 Rvir."""
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



print("Finding potential.")
potential = Parallel(n_jobs=NTHREADS)(delayed(find_potential)(i)
                                      for i in range(halos.size))
