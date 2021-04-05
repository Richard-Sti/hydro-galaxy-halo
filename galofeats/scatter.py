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

"""A tool to calculate the Gaussian scatter in any 1D relation."""

import numpy
from scipy.signal import savgol_filter


class ScatterEstimator:
    """
    A class to estimate the Gaussian scatter of 1D target `y` as a
    function of 1D feature `x`.

    The scatter is estimated by inspecting nearby points within a predefined
    bin width and delete-d jackknife resampling the points within the bin.
    Calculates the mean position within the bin, scatter within the bin,
    and uncertainty on the scatter.


    Parameters
    ----------
    x : numpy.ndarray (1D)
        Feature array.
    y : numpy.ndarray (1D)
        Target array.
    """

    def __init__(self, x, y, bin_width):
        self._x = None
        self._y = None
        self._half_dx = bin_width / 2
        self.x = x
        self.y = y

    @property
    def x(self):
        """Feature array."""
        return self._x

    @x.setter
    def x(self, x):
        """Sets `x`."""
        if x.ndim != 1:
            raise ValueError("'x' must be a 1D numpy aray.")
        self._x = x

    @property
    def y(self):
        """Target array."""
        return self._y

    @y.setter
    def y(self, y):
        """Sets `y`."""
        if y.ndim != 1:
            raise ValueError("'y' must be a 1D numpy aray.")
        self._y = y

    @property
    def half_dx(self):
        """
        Returns the half bin-width :math:`dx` on `x`. Given a pivot point
        :math:`\phi`, observations :math:`x` that fall within

        .. math::

            \phi - dx < x < \phi + dx

        are used to estimate the local scatter.
        """
        return self._half_dx

    def point_stats(self, phi):
        """
        Jackknife estimate of the mean `y` and scatter on `x` at `phi`.

        Parameters
        ----------
        phi : float
            Point at which to estimate the scatter.

        Returns
        -------
        result : dict
            Dictionary with keys `x`, `y`, `scatter`, and `scatter_std`.
        """
        # Points within the bin width of phi
        mask = numpy.logical_and(self.x > phi - self.half_dx,
                                 self.x < phi + self.half_dx)
        xbin = self.x[mask]
        ybin = self.y[mask]
        # Number of points within the bin
        Nbin = xbin.size
        # Number of jack samples
        Njack = int(Nbin**0.5)
        # Randomly split the bin points among Njack groups
        groups = numpy.random.randint(0, Njack, size=Nbin)
        # Eliminate a group at a time and calculate the statistics
        stat = numpy.zeros(shape=(Njack, 3))
        for i in range(Njack):
            jack_mask = groups != i
            stat[i, :] = (numpy.mean(xbin[jack_mask]),
                          numpy.mean(ybin[jack_mask]),
                          numpy.std(ybin[jack_mask]))
        # Calculate the mean of the jackknife samples
        xmu = numpy.mean(stat[:, 0])
        ymu = numpy.mean(stat[:, 1])
        scatter = numpy.mean(stat[:, 2])

        # Calculate the jackknife standard deviation
        # Number of observations within each group
        Ngroups = numpy.array([(groups == i).sum() for i in range(Njack)])
        # Jackknife proportionality constant for each jackknife sample
        C = (Nbin - Ngroups) / Nbin
        scatter_std = numpy.sum(C * (stat[:, 2] - scatter)**2)**0.5

        return {'x': xmu,
                'y': ymu,
                'scatter': scatter,
                'scatter_std': scatter_std}

    def __call__(self, knots, **kwargs):
        """
        Calculates the jackknife statistics at positions `knots`.

        Parameters
        ----------
        knots : numpy.ndarray (1D)
            Position along `x` where to estimate the scatter.
         **kwargs :
             Optional keyword arguments passed into
             :py:func:`scipy.signal.savgol_filter` to smooth the output mean
             and scatter curves. By default, if no arguments are provided,
             the results are not smoothed.

        Returns
        -------
        result : structured numpy.ndarray
            A structured array with named fields `x`, `y`, `scatter`,
            `scatter_std`.
        """
        stats = [self.point_stats(knot) for knot in knots]
        # Turn the list of dictionaries into a numpy structured array
        attrs = [key for key in stats[0].keys()]
        X = numpy.zeros(len(stats), dtype={'names': attrs,
                                           'formats': ['float64']*len(attrs)})
        for i, stat in enumerate(stats):
            for key, value in stat.items():
                X[i][key] = value
        # Optionally apply savgol filter to smooth the output
        if len(kwargs) != 0:
            for attr in attrs:
                X[attr] = savgol_filter(X[attr], **kwargs)
        return X
