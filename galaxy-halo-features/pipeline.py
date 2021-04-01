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

"""A union pipeline to combine multiple preprocessing pipelines."""

import numpy
from sklearn.base import (BaseEstimator, TransformerMixin)
from sklearn.model_selection import StratifiedShuffleSplit


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
        Attributes to which log transform is applied. By default None.
    """

    def __init__(self, attributes, log_attributes=None):
        self._attributes = None
        self._log_attributes = None

        self.attributes = attributes
        self.log_attributes = log_attributes

    @staticmethod
    def _enforce_list_str(strings, name):
        """A support function to ensure `strings` is a list of strings."""
        if isinstance(strings, tuple):
            attributes = list(strings)
        if not isinstance(strings, (list, str)):
            raise ValueError("'{}' must be a list or a single string"
                             .format(name))
        if isinstance(strings, str):
            strings = [strings]
        for attr in strings:
            if not isinstance(attr, str):
                raise ValueError("{} '{}' must be a string"
                                 .format(name, attr))
        return strings

    @property
    def attributes(self):
        """Attributes handled by this selector."""
        if self._attributes is None:
            raise ValueError("'attributes' not set.")
        return self._attributes

    @attributes.setter
    def attributes(self, attributes):
        """Sets the attributes."""
        self._attributes = self._enforce_list_str(attributes, 'attribute')

    @property
    def log_attributes(self):
        """Returns the attributes which are to be log-transformed."""
        if self._log_attributes is None:
            return []
        return self._log_attributes

    @log_attributes.setter
    def log_attributes(self, attributes):
        """Sets the log attributes."""
        if attributes is None:
            return
        attributes = self._enforce_list_str(attributes, 'log_attribute')
        # Check that each attribute is in `attributes`
        for attr in attributes:
            if attr not in self.attributes:
                raise ValueError("Log attribute '{}' not found in attributes"
                                 .format(attr))
        self._log_attributes = attributes

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
                out[i] = numpy.log10(X[par])
            else:
                out[i] = X[par]
        return numpy.vstack(out).T

    def inverse_transform(self, X):
        """Inverse transforms the data frame."""
        out = numpy.zeros(
                X.shape[0],
                dtype={'names': self.attributes,
                       'formats': ['float64'] * len(self.attributes)})
        for i, par in enumerate(self.attributes):
            if par in self.log_attributes:
                X[:, i] = 10**X[:, i]
            out[par] = X[:, i]
        return out


class UnionPipeline:
    """
    A union pipeline of preprocessing pipelines.

    Parameters
    ----------
    pipelines : list of sklearn.pipeline.Pipeline
        The individual pipelines.
    """

    def __init__(self, pipelines):
        self._attributes = []
        self._pipelines = None
        self.pipelines = pipelines

    @property
    def pipelines(self):
        """Individual pipelines handled by this union pipeline."""
        return self._pipelines

    @pipelines.setter
    def pipelines(self, pipelines):
        """Sets the pipelines. Checks for correct input."""
        if not isinstance(pipelines, list):
            raise ValueError("'pipelines' must be a list.")
        for pipeline in pipelines:
            if not isinstance(pipeline, Pipeline):
                raise ValueError("Pipeline '{}' must be of "
                                 "sklearn.pipeline.Pipeline".format(pipeline))
        self._pipelines = pipelines

    @property
    def attributes(self):
        """Attributes handled by this union pipeline."""
        if self._attributes is None:
            raise ValueError("'attributes' not set. The pipeline must be "
                             "fitted first.")
        return self._attributes

    def _prepare_pipeline_transform(self, X, pipeline, names):
        """
        Support function that extracts `pipeline` specific columns from `X`
        and removes the attribute names from `names`.
        """
        if pipeline.steps[0][0] != 'selector':
            raise ValueError("The pipeline's first step must be "
                             "a selector.")
        attributes = pipeline.steps[0][1].attributes
        # Create the structured array that will be given to the pipeline
        out = numpy.zeros(X.shape[0],
                          dtype={'names': attributes,
                                 'formats': ['float64'] * len(attributes)})
        # Pop the pipeline's attributes from the name list
        for attr in attributes:
            if attr not in names:
                raise ValueError("Pipeline attribute '{}' not found in X"
                                 .format(attr))
            names.remove(attr)
            out[attr] = X[attr]
        return out

    def fit_transform(self, X):
        """
        Fits the individual transforms and returns the transformed data.

        Parameters
        ----------
        X : numpy.ndarray with named fields
            Data to be transformed. Must have a specified pipeline.
        """
        names = list(X.dtype.names)
        out = [None] * len(self.pipelines)

        for i, pipeline in enumerate(self.pipelines):
            arr = self._prepare_pipeline_transform(X, pipeline, names)
            out[i] = pipeline.fit_transform(arr)
            # Save the attribute order. Will be used for the inverse transf.
            for attr in pipeline.steps[0][1].attributes:
                self._attributes.append(attr)

        if len(names) != 0:
            raise ValueError("Features {} were not fitted and transformed."
                             .format(names))
        return numpy.hstack(out)

    def transform(self, X):
        """
        Transforms data specified by `X`.

        Parameters
        ----------
        X : numpy.ndarray with named fields
            Data to be transformed. Must have a specified pipeline.
        """
        names = list(X.dtype.names)
        out = [None] * len(self.pipelines)

        for i, pipeline in enumerate(self.pipelines):
            arr = self._prepare_pipeline_transform(X, pipeline, names)
            out[i] = pipeline.transform(arr)

        if len(names) != 0:
            raise ValueError("Features {} were not transformed.".format(names))
        return numpy.hstack(out)

    def inverse_transform(self, X):
        """
        Inverse transforms the data. Returns a numpy structured array
        with the original data.
        """
        out = numpy.zeros(
                X.shape[0],
                dtype={'names': self.attributes,
                       'formats': ['float64'] * len(self.attributes)})
        start = 0
        for i, pipeline in enumerate(self.pipelines):
            attribs = pipeline.steps[0][1].attributes

            end = start + len(attribs)
            inv = pipeline.inverse_transform(X[:, start:end])
            for par in inv.dtype.names:
                out[par] = inv[par]
            # Bump up the counter
            start += end
        return out


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
        axis = numpy.log10(axis)

    axmin, axmax = [numpy.sort(axis)[int(p * axis.size)]
                    for p in ax_percentile]
    bands = numpy.digitize(axis, numpy.linspace(axmin, axmax, Nbins))

    # Unpack the data into smaller structured arrays
    X = numpy.zeros(axis.size, dtype={'names': features,
                                      'formats': ['float64'] * len(features)})
    for feat in features:
        X[feat] = data[feat]

    y = numpy.zeros(axis.size, dtype={'names': [target],
                                      'formats': ['float64']})
    y[target] = data[target]

    # Perform the stratify split
    split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,
                                   random_state=seed)
    for train_index, test_index in split.split(X, bands):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test
