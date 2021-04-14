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

"""Regressor class that provides an overlay to fit and inspect estimators."""

from abc import (ABC, abstractmethod)

import numpy
from sklearn import metrics
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve
from sklearn.base import clone
from joblib import (parallel_backend, Parallel, delayed)

from .preprocess import UnionPipeline


class BaseRegressor(ABC):
    """Base regressor class.

    Attributes
    ----------
    _X_train
    _X_test
    _y_train
    _y_test
    _feature_pipeline
    _target_pipeline
    _is_fitted
    """

    _X_train = None
    _X_test = None
    _y_train = None
    _y_test = None
    _feature_pipeline = None
    _target_pipeline = None
    _is_fitted = False

    @property
    def X_train(self):
        """Training features."""
        if self._X_train is None:
            raise ValueError("'X_train' not set.")
        return self._X_train

    @X_train.setter
    def X_train(self, X):
        """Sets the training features."""
        if not (isinstance(X, numpy.ndarray) and X.dtype.names is not None):
            raise ValueError("'X_train' must be a numpy structured array "
                             "with named fields.")
        self._X_train = X

    @property
    def X_test(self):
        """Test features."""
        if self._X_test is None:
            raise ValueError("'X_test' not set.")
        return self._X_test

    @X_test.setter
    def X_test(self, X):
        """Sets the test features. Must be set after `X_train`."""
        if not (isinstance(X, numpy.ndarray) and X.dtype.names is not None):
            raise ValueError("'X_train' must be a numpy structured array "
                             "with named fields.")

        if X.dtype.names != self.X_train.dtype.names:
            raise ValueError("Named fields of 'X_test' differ from 'X_train'.")
        self._X_test = X

    @property
    def y_train(self):
        """Training target."""
        if self._y_train is None:
            raise ValueError("'y_train' not set.")
        return self._y_train

    @y_train.setter
    def y_train(self, y):
        """Sets the training features."""
        if not (isinstance(y, numpy.ndarray) and y.dtype.names is not None):
            raise ValueError("'y_train' must be a numpy structured array "
                             "with named fields.")
        self._y_train = y

    @property
    def y_test(self):
        """Test target."""
        if self._y_test is None:
            raise ValueError("'y_test' not set.")
        return self._y_test

    @y_test.setter
    def y_test(self, y):
        """Sets the test target. Must be set after `y_train`."""
        if not (isinstance(y, numpy.ndarray) and y.dtype.names is not None):
            raise ValueError("'y_train' must be a numpy structured array "
                             "with named fields.")

        if y.dtype.names != self.y_train.dtype.names:
            raise ValueError("Named fields of 'y_test' differ from 'y_train'.")
        self._y_test = y

    @property
    def feature_pipeline(self):
        """Feature preprocessing pipeline."""
        return self._feature_pipeline

    @feature_pipeline.setter
    def feature_pipeline(self, pipe):
        """Sets the feature pipeline."""
        if not isinstance(pipe, UnionPipeline):
            raise ValueError("Invalid pipeline type '{}'".format(type))
        self._feature_pipeline = pipe

    @property
    def target_pipeline(self):
        """Target preprocessing pipeline."""
        return self._target_pipeline

    @target_pipeline.setter
    def target_pipeline(self, pipe):
        """Sets the target pipeline."""
        if not isinstance(pipe, UnionPipeline):
            raise ValueError("Invalid pipeline type '{}'".format(type))
        self._target_pipeline = pipe

    @property
    def is_fitted(self):
        """Whether the grid or model is fitted."""
        return self._is_fitted

    @abstractmethod
    def fit(self, **kwargs):
        """Fits the model."""
        pass

    @abstractmethod
    def predict(self, X):
        """Predicts target values using the fitted model."""
        pass

    @abstractmethod
    def score(self, **kwargs):
        """Scores the model... improve docs."""
        pass


class SklearnRegressor(BaseRegressor):
    """
    Overlay for sklearn regressors to fit, score, and predict targets
    either using a parameter grid or a single estimator.

    Parameters
    ----------
    grid : sklearn grid or estimator object.
        Hyperparameter estimator grid to be explored or a single estimator
        to be fitted.
    X_train : `numpy.ndarray` with named fields
        Training features.
    X_test : `numpy.ndarray` with named fields
        Test features.
    y_train : `numpy.ndarray` with named fields
        Training targets.
    y_test : `numpy.ndarray` with named fields
        Test targets.
    feature_pipeline : `galofeats.UnionPipeline`
        Preprocessing pipeline handling forward and inverse transforms
        of features.
    target_pipeline : `galofeats.UnionPipeline`
        Preprocessing pipeline handling forward and inverse transforms
        of targets.
    """

    def __init__(self, model, X_train, X_test, y_train, y_test,
                 feature_pipeline, target_pipeline):
        # Decide whether a grid or a single estimator was passed
        if 'model_selection' in str(type(model)):
            self._grid = model
            self._model = None
        else:
            self._grid = None
            self._model = model

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_pipeline = feature_pipeline
        self.target_pipeline = target_pipeline

        # Add support for weights on this level ?

    @property
    def model(self):
        """Returns the best scoring grid estimator. If a single estimator
        is passed, instead of a grid, simply returns the estimator."""
        if self._model is None:
            try:
                return self.grid.best_estimator_
            except AttributeError:
                raise Exception("`grid` must be fitted first.")
        return self._model

    @property
    def grid(self):
        """The regressor's parameter grid."""
        if self._grid is None:
            raise ValueError("'grid' is not set.")
        return self._grid

    def fit(self, **kwargs):
        """
        Fits the grid's estimators or a single estimator if `model` is passed
        instead.

        Parameters
        ----------
            **kwargs :
                Optional arguments passed into `self.model.fit`.
        Returns
        -------
        None
        """
        X = self.feature_pipeline.fit_transform(self.X_train)
        y = self.target_pipeline.fit_transform(self.y_train)
        # Some models prefer 1-dimensional arrays if a single target.
        if len(self.target_pipeline.attributes) == 1:
            y = y.reshape(-1,)
        # Check if any kwargs
        if len(kwargs) == 0:
            if self._model is None:
                self.grid.fit(X, y)
            else:
                self.model.fit(X, y)
        else:
            if self._model is None:
                self.grid.fit(X, y, kwargs)
            else:
                self.model.fit(X, y, kwargs)
        self._is_fitted = True

    def predict(self, X=None):
        """
        Predicts the target values using the grid's best estimator or the
        single estimator if `model` is passed instead. Applies the inverse
        transform.

        Parameters
        ----------
            X : numpy.ndarray, optional
                Optionally can predict target values for sets other than the
                test set. Must have the same structure as `self.X_train`.
                The regressor will automatically scale the data.
        Returns
        -------
        result : numpy.ndarray with named fields
            Predicted target variables for the test set or `X` if passed.
        """
        if X is not None:
            if not isinstance(X, numpy.ndarray):
                raise ValueError("'X' must be of numpy.ndarray type.")
            if X.dtype.names != self.X_train.dtype.names:
                raise ValueError("Named fields of 'X' do not match "
                                 "`self.X_train`.")
            X = self.feature_pipeline.transform(X)
        else:
            X = self.feature_pipeline.transform(self.X_test)

        y = self.model.predict(X)
        # Some models return 1D arrays
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return self.target_pipeline.inverse_transform(y)

    def score(self, **kwargs):
        """
        Calculates the R2, mean absolute error, and explained variance
        scores on the transformed target data using either the grid's best
        estimator or using the single estimator if `model` is passed instead.

        Parameters
        ----------
            **kwargs :
                Optional arguments passed into `sklearn.metrics` scorers.
        Returns
        -------
        score : dict
            Dictionary with keys: `R2`, `MAE`, and `explained_variance`.
        """
        # Calculate the predicted values
        X = self.feature_pipeline.transform(self.X_test)
        ypred = self.model.predict(X)
        # Transform the target set
        ytest = self.target_pipeline.transform(self.y_test)

        if len(kwargs) == 0:
            r2 = metrics.r2_score(ytest, ypred)
            mae = metrics.mean_absolute_error(ytest, ypred)
            exp_var = metrics.explained_variance_score(ytest, ypred)
        else:
            r2 = metrics.r2_score(ytest, ypred, kwargs)
            mae = metrics.mean_absolute_error(ytest, ypred, kwargs)
            exp_var = metrics.explained_variance_score(ytest, ypred, kwargs)

        score = {'R2': r2,
                 'MAE': mae,
                 'explained_varince': exp_var}
        return score

    def permutation_importance(self, scoring=None, n_jobs=None, n_repeats=10,
                               seed=42):
        """
        Calculates the normalised permutation importances on the test set
        such that the sum is 1.

        Parameters
        ----------
        scoring : string, callable, optional
            The scorer to use. It can either be a single string (supported
            by sklearn) or a sklearn callable metric. By default uses the
            estimator's default scorer.

            For more details see:
                https://scikit-learn.org/stable/modules/model_evaluation.html
        n_jobs : int, optional
            Number of jobs to run in parallel. By default 1.
        n_repeats : int, optional
            Number of times to permuate a feature. By default 10.
        seed : int, optional
            Random state.

        Returns
        -------
        result : numpy.ndarray
            Array of shape (n_features, 2). The columns represent the mean
            and standard deviation of the permutation importances,
            respectively.
        """
        if not self.is_fitted:
            raise RuntimeError("The grid or model must be fitted first.")
        X = self.feature_pipeline.transform(self.X_test)
        y = self.target_pipeline.transform(self.y_test)

        perm = permutation_importance(estimator=self.model, X=X, y=y,
                                      scoring=scoring, n_repeats=n_repeats,
                                      n_jobs=n_jobs, random_state=seed,
                                      sample_weight=None)

        out = numpy.vstack([perm.importances_mean, perm.importances_std]).T
        out /= numpy.sum(out[:, 0])
        return out

    def feature_importance(self, n_repeats=10, n_jobs=1, seed=42):
        """
        Calculates feature importances on the training set for ensemble
        models. For invalid models returns `None`.

        Parameters
        ----------
        n_jobs : int, optional
            Number of jobs to run in parallel. By default 1.
        n_repeats : int, optional
            Number of times to refit the model. By default 10.
        seed : int, optional
            Random state.

        Returns
        -------
        result : numpy.ndarray
            Array of shape (n_features, 2). The columns represent the mean
            and standard deviation of the feature importances, respectively.
        """
        if not self.is_fitted:
            raise RuntimeError("The grid or model must be fitted first.")
        try:
            self.model.feature_importances_
        except AttributeError:
            return None

        seeds = numpy.random.randint(0, 2**32, size=n_repeats)
        X = self.feature_pipeline.transform(self.X_train)
        y = self.target_pipeline.transform(self.y_train)

        if len(self.target_pipeline.attributes) == 1:
            y = y.reshape(-1,)

        with parallel_backend('loky', n_jobs=n_jobs):
            out = Parallel()(delayed(self._feat_fit)(X, y, seed)
                             for seed in seeds)
        # Massage the backend output a little bit
        out = numpy.vstack(out)
        mu = numpy.mean(out, axis=0)
        std = numpy.std(out, axis=0)
        # We want to output an aray
        return numpy.vstack([mu, std]).T

    def _feat_fit(self, X, y, seed, weight=None):
        """
        A support function for `self.feature_importance` method. Refits the
        best-fit ensemble estimator and returns the feature importances.
        """
        # Clone the best estimator. But still have to change the random seed
        model = clone(self.model)
        model.random_state = seed
        model.fit(X, y, sample_weight=weight)
        return model.feature_importances_

    def learning_curve(self, train_sizes=None, cv=None, scoring=None,
                       n_jobs=1, seed=42, **kwargs):
        """
        Calculates the cross-validated learning curve on the training set
        using the best-scoring estimator.

        Parameters
        ----------
        train_sizes : list, optional
            Relative sizes of the training set used to calculate the learning
            curve. By default [0.1, 0.33, 0.66, 1.0]
        cv : int, optional
            Number of folds for cross-validation. By default 5.
        scoring : string, callable, optional
            The scorer to use. It can either be a single string (supported
            by sklearn) or a sklearn callable metric. By default uses the
            estimator's default scorer.

            For more details see:
                https://scikit-learn.org/stable/modules/model_evaluation.html
        n_jobs : int, optional
            Number of jobs to run in parallel. By default 1.
        n_repeats : int, optional
            Number of times to permuate a feature. By default 10.
        seed : int, optional
            Random state.
        **kwargs :
            Optional arguments passed into the fit method of the estimator.

        Returns
        -------
        result : dict
            A dictionary with keys `train_sizes`, `train_scores`,
            `test_scores`, `fit_times`, `score_times`. Where applicable
            the specific output is of shape (`len(train_sizes)`, `cv`).
        """
        if train_sizes is None:
            train_sizes = [0.1, 0.33, 0.66, 1.0]
        X = self.feature_pipeline.transform(self.X_train)
        y = self.target_pipeline.transform(self.y_train)
        # Calculate the learning curve
        curve = learning_curve(self.model, X, y, train_sizes=train_sizes,
                               cv=cv, scoring=scoring, n_jobs=n_jobs,
                               shuffle=True, random_state=seed,
                               return_times=True, fit_params=kwargs)
        # Unpack the curve's output
        keys = ['train_sizes', 'train_scores', 'test_scores', 'fit_times',
                'score_times']
        return {keys[i]: res for i, res in enumerate(curve)}
