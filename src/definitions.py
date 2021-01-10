"""Contains some basic functions."""
import numpy as np

from astropy import units as u


def Rvir_from_mvir(mvir, cosmo, z=0):
    """
    Calculates the halo virial radius, assuming a flat universe.

    Note the choice of overdensity definition. Assumes ``mvir`` is initially
    in solar masses.
    """
    # Calculate the halo virial radius
    mvir = mvir.copy() * u.Msun / (cosmo.H0.to_value()/100)
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
