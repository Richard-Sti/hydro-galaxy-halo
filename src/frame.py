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

"""A frame for a simple data manipulation."""

from abc import (ABC, abstractmethod)

import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import (BaseEstimator, TransformerMixin)



class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    A simple class to convert data from a numpy structured array into a
    sklearn-friendly format.

    Optionally applies :math:`\log_{10}` to the specified attributes.

    Parameters
    ----------
    attributes : (list of) str
        Attributes to be extracted from the structured array.
    log_attributes : (list of) str, optional
        Attributes to which log transform is applied.
    """

    def __init__(self, attributes, log_attributes):
        self.attributes = attributes
        self.log_attributes = log_attributes
        # Ensure that by defauult no LOG!

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Unpacks the data frame. Takes logarithmic transforms of beforehand
        specified parameters.
        """
        out = [None] * len(self.attributes)
        for i, par in enumerate(self.attributes):
            if par in self.log_attributes:
                out[i] = np.log10(X[par])
            else:
                out[i] = X[par]
        return np.vstack(out).T

    def inverse_transform(self, X):
        """Inverse transforms the data frame."""
        out = np.zeros(X.shape[0],
                       dtype={'names': self.attributes,
                              'formats': ['float64'] * len(self.attributes)})
        for i, par in enumerate(self.attributes):
            if par in self.log_attributes:
                X[:, i] = 10**X[:, i]
            out[par] = X[:, i]
        return out


class UnionPipeline:
    """
    P
    """

    def __init__(self, pipelines):
        # Check how pipelines are inputted
        pass

    def fit_transform(self, X):
        """
        Fits the individual transforms.

        Parameters
        ----------
        X : numpy.ndarray with named fields
            Data to be transformed. Must have a specified pipeline.
        """
        for pipeline in pipelines:
            attributes = pipeline.steps[0][1].attributes
            # Always ensure that selector is the first thing in the pipeline!
            # Ensure that the attributes match the data array
            # Ensure that after transforming all columns of X have been transformed!

            # Create a numpy array of the desired shape using the given attributes
            # Transform this array and fit the thing
            # Cache the result and at the end return everything at once


    def transfrom(self, X):
        pass

    def inverse_transform(self, X):
        pass



# Throw this ugly mess away


# class Frame:
# 
#     def __init__(self, features, target, subhalos, split='stratify',
#                  test_size=0.2, dy=0.2, seed=42):
#         self.features_pipe = None
#         self.target_pipe = None
#         self.labels = [p for p in features.keys()]
#         self.target = target['label']
#         self.subhalos = subhalos
# 
#         X, y = self.get_raw_data(features, target, subhalos)
#         X_train, X_test, y_train, y_test = self.split_data(
#                 X, y, split, test_size, dy, seed)
# 
#         # Apply transformations
#         self.fit_scaler(X_train, y_train, features, target)
#         self.X_train = self.features_pipe.transform(X_train)
#         self.X_test = self.features_pipe.transform(X_test)
#         self.y_train = self.target_pipe.transform(y_train).reshape(-1,)
#         self.y_test = self.target_pipe.transform(y_test).reshape(-1,)
# 
#     def get_raw_data(self, features, target, subhalos):
#         """Loads the raw data and applies log10 transform if any specified."""
#         # Loads the matched data
#         match = np.load('/mnt/zfsusers/rstiskalek/hydro/data/matched.npy')
#         if subhalos not in [0, 1, 2]:
#             raise ValueError("Invalid handle: 'subhalos': {}".format(subhalos))
#         if subhalos == 0:
#             mask = np.ones_like(match['level'], dtype=bool)
#         else:
#             mask = match['level'] == subhalos
# 
#         # Get the target
#         y = match[mask][target['label']].reshape(-1, 1)
#         if 'log10' in target['transforms']:
#             y = np.log10(y)
# 
#         # Get the features
#         X = [None] * len(features)
#         for i, (key, value) in enumerate(features.items()):
#             X[i] = match[mask][key]
#             if 'log10' in value:
#                 X[i] = np.log10(X[i])
#         X = np.vstack(X).T
#         return X, y
# 
#     def split_data(self, X, y, split, test_size, dy, seed):
#         """Splits the data."""
#         y_sorted = np.sort(y.reshape(-1))
#         N = len(y_sorted)
#         ymin = y_sorted[int(0.01 * N)]
#         ymax = y_sorted[int(0.99 * N)]
#         bins = np.arange(ymin, ymax, dy)
# 
#         bands = np.digitize(y, bins)
# 
#         sss = StratifiedShuffleSplit(n_splits=50, test_size=test_size,
#                                      random_state=seed)
#         for train_index, test_index in sss.split(X, bands):
#             X_train, X_test = X[train_index], X[test_index]
#             y_train, y_test = y[train_index], y[test_index]
#         return X_train, X_test, y_train, y_test
# 
#     def fit_scaler(self, X_train, y_train, features, target):
#         """Pass"""
#         cols_standard = []
#         cols_minmax = []
#         for i, (key, value) in enumerate(features.items()):
#             if 'standard' in value:
#                 cols_standard.append(i)
#             elif 'minmax' in value:
#                 cols_minmax.append(i)
# 
#         self.features_pipe = ColumnTransformer(
#                 [('standard', StandardScaler(), cols_standard),
#                  ('minmax', MinMaxScaler(), cols_minmax)],
#                 remainder='passthrough')
# 
#         if 'standard' in target['transforms']:
#             self.target_pipe = StandardScaler()
#         else:
#             raise ValueError('Unsupported target transform.')
# 
#         self.features_pipe.fit_transform(X_train)
#         self.target_pipe.fit_transform(y_train)
