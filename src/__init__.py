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

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

#
# =============================================================================
#
#                     Stratify split utilities
#
# =============================================================================
#


def stratify_split(data, features, target, log_target, test_size=0.2, seed=42,
                   Nbins=10, n_splits=50, ax_percentile=(0.01, 0.99)):
    """
    Performs stratified split on the input data structured array and
    return X_train, X_test, y_train, and y_test structured arrats.

    The stratified split is performed along the target axis.

    Parameters
    ----------
    data : numpy.ndarray with named field
        Input structured data array that contains both the features and
        target variable.
    features : list of str
        Attributes that will
    target : str
        Target attribute
    log_target : bool
        Whether to log-transform the target axis before performing the
        stratified split.
    test_size : float, optional
        A fractional size of the test set. Must be larger than 0 and smaller
        than 1. By default 0.2.
    seed : int
        Random seed.
    Nbins : int, optional
        Number of bins for stratified splitting. By default 10.
    n_splits : int, optional
        Number of re-shuffling & splitting iterations. By default 50.
    ax_percentile: len-2 tuple, optional
        Percentile range to estimate the bins for stratified split.
        By default (0.01, 0.99).

    Returns
    ------
    X_train : numpy.ndarray
        Train features.
    X_test : numpy.ndarray
        Test features.
    y_train : numpy.ndarray
        Train target.
    y_test : numpy.ndarray
        Test target.
    """

    # Check the features inputs
    if not isinstance(features, list):
        raise ValueError("'features' must be a list.")
    for feat in features:
        if not isinstance(feat, str):
            raise ValueError("Feature '{}' must be a string.".format(feat))
    # Check the target feature
    if not isinstance(target, str):
        raise ValueError("'target' must be a string. Only a single target "
                         "feature is allowed.")
    # And check the other inputs..
    if not isinstance(log_target, bool):
        raise ValueError("'log_target' must be a boolean.")
    for p, v in zip(['Nbins', 'seed', 'n_splits'], [Nbins, seed, n_splits]):
        if not isinstance(v, int):
            raise ValueError("'{}' must be an integer.".format(p))
    if not 0.0 < test_size < 1.0:
        raise ValueError("'test_size' must be between 0 and 1.")
    if not (isinstance(ax_percentile, (list, tuple))
            and len(ax_percentile) == 2):
        raise ValueError("'ax_percentile' must be a len-2 list or tuple.")
    ax_percentile = list(ax_percentile)
    for val in ax_percentile:
        if not 0.0 < val < 1.:
            raise ValueError("'ax_percentile' must be between 0 and 1.")
    # Enforce an increasing order
    if not ax_percentile[1] > ax_percentile[0]:
        ax_percentile = ax_percentile[::-1]
    # Stratify axis
    axis = data[target]
    if log_target:
        axis = np.log10(axis)

    axmin, axmax = [np.sort(axis)[int(p * axis.size)] for p in ax_percentile]
    bands = np.digitize(axis, np.linspace(axmin, axmax, Nbins))

    # Unpack the data into smaller structured arrays
    X = np.zeros(axis.size, dtype={'names': features,
                                   'formats': ['float64'] * len(features)})
    for feat in features:
        X[feat] = data[feat]

    y = np.zeros(axis.size, dtype={'names': [target], 'formats': ['float64']})
    y[target] = data[target]

    # Perform the stratify split
    split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,
                                   random_state=seed)
    for train_index, test_index in split.split(X, bands):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test
