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

"""Pipeline for sklearn models"""

from abc import (ABC, abstractmethod)

import numpy

from sklearn import metrics
from sklearn.inspection import permutation_importance
from joblib import parallel_backend


from sklearn.pipeline import Pipeline
from .preprocess import UnionPipeline


class BaseRegressor(ABC):
    """Base regressor class.

    Attributes

    Add docs!

    """
    _X_train = None
    _X_test = None
    _y_train = None
    _y_test = None
    _feature_pipeline = None
    _target_pipeline = None

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

    @abstractmethod
    def fit(self, **kwargs):
        """Fits the model."""
        pass

    @abstractmethod
    def predict(self, X_test):
        """Predicts target values using the fitted model."""
        pass

    @abstractmethod
    def score(self, **kwargs):
        """Scores the model... improve docs."""
        pass


class SklearnModelRegressor(BaseRegressor):
    """
    Handles scaling internally!!!


    """

    def __init__(self, model, X_train, X_test, y_train, y_test,
                 feature_pipeline, target_pipeline):
        self._model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_pipeline = feature_pipeline
        self.target_pipeline = target_pipeline

    @property
    def model(self):
        """The sklearn model used by this regressor."""
        if self._model is None:
            raise ValueError("'model' is not set.")
        return self._model


    def fit(self, **kwargs):
        """
        Fits the model.

        Parameters
        ----------
            **kwargs :
                Optional arguments passed into `self.model.fit`.
        """
        X = self.feature_pipeline.fit_transform(self.X_train)
        y = self.target_pipeline.fit_transform(self.y_train)
        # Some models prefer 1-dimensional arrays if a single target.
        if len(self.target_pipeline.attributes) == 1:
            y = y.reshape(-1,)
        # Check if any kwargs
        if len(kwargs) == 0:
            self.model.fit(X, y)
        else:
            self.model.fit(X, y, kwargs)


    def predict(self, X=None):
        """
        Predicts the target values using the fitted model. Applies the inverse
        transform.

        Parameters
        ----------
            X : numpy.ndarray, optional
                Optionally can predict target values for sets other than the
                test set. Must have the same structure as `self.X_train`.
                The regressor will automatically scale the data.
        """
        if X is not None:
            if not isinstance(X, numpy.ndarray):
                raise ValueError("'X' must be of numpy.ndarray type.")
            if X.dtype.names != self.X_train.dtype:
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
        scores on the transformed target data.

        Parameters
        ----------
            **kwargs :
                Optional arguments passed into `sklearn.metrics` scorers.
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


class SklearnGridRegressor(BaseRegressor):
    """
    THIS IS JUST A COPY OF THE MODEL REGRESSOR REDO THIS.

    BE CAREFUL HOW TO PASS PARAMS INTO THE CV GRID

    CV GRID - OBTAIN FEATURE IMPORTANCE FOR THE FEW TOP MODELS?


    """

    def __init__(self, grid, X_train, X_test, y_train, y_test,
                 feature_pipeline, target_pipeline):
        self._grid = grid
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_pipeline = feature_pipeline
        self.target_pipeline = target_pipeline

    @property
    def model(self):
        """The sklearn model used by this regressor."""
        if self._model is None:
            raise ValueError("'model' is not set.")
        return self._model


    def fit(self, **kwargs):
        """
        Fits the model.

        Parameters
        ----------
            **kwargs :
                Optional arguments passed into `self.model.fit`.
        """
        X = self.feature_pipeline.fit_transform(self.X_train)
        y = self.target_pipeline.fit_transform(self.y_train)
        # Some models prefer 1-dimensional arrays if a single target.
        if len(self.target_pipeline.attributes) == 1:
            y = y.reshape(-1,)
        # Check if any kwargs
        if len(kwargs) == 0:
            self.model.fit(X, y)
        else:
            self.model.fit(X, y, kwargs)


    def predict(self, X=None):
        """
        Predicts the target values using the fitted model. Applies the inverse
        transform.

        Parameters
        ----------
            X : numpy.ndarray, optional
                Optionally can predict target values for sets other than the
                test set. Must have the same structure as `self.X_train`.
                The regressor will automatically scale the data.
        """
        if X is not None:
            if not isinstance(X, numpy.ndarray):
                raise ValueError("'X' must be of numpy.ndarray type.")
            if X.dtype.names != self.X_train.dtype:
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
        scores on the transformed target data.

        Parameters
        ----------
            **kwargs :
                Optional arguments passed into `sklearn.metrics` scorers.
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





# class RegressionModel:
#     """A simple regression model."""
# 
#     def __init__(self, model, name, parameter_grid, frame, search='random',
#                  n_iters=10, n_jobs=1, verbose=3, seed=42, cv=5,
#                  perm_repeats=25):
#         self.name = name
#         self.n_jobs = n_jobs
#         self.frame = frame
#         self.seed = seed
#         self.perm_repeats = perm_repeats
#         self.cv = cv
#         # Initialise the grids
#         if search == 'random':
#             self.grid = RandomizedSearchCV(model, parameter_grid,
#                                            n_iter=n_iters,
#                                            scoring='neg_mean_absolute_error',
#                                            cv=cv, verbose=verbose)
#         elif search == 'grid':
#             self.grid = GridSearchCV(model, parameter_grid,
#                                      scoring='neg_mean_absolute_error', cv=cv,
#                                      verbose=verbose)
#         else:
#             raise ValueError("Invalid search option")
# 
#     def fit_grid(self):
#         """Fits the model."""
#         with parallel_backend('loky', n_jobs=self.n_jobs):
#             self.grid.fit(self.frame.X_train, self.frame.y_train)
#         # Append the column names
#         self.grid.cv_results_.update({'columns': self.frame.labels})
#         self.grid.cv_results_.update({'target': self.frame.target})
# 
#     def score_model(self, print_results=True):
#         """Scores the models on the test data."""
#         y_pred = self.grid.best_estimator_.predict(self.frame.X_test)
# 
#         # Transform back
#         inv = self.frame.target_pipe.inverse_transform
# 
#         test_error = mean_absolute_error(self.frame.y_test, y_pred)
#         test_error_inv = mean_absolute_error(inv(self.frame.y_test),
#                                              inv(y_pred))
#         # Append the test batch error and the test data
#         self.grid.cv_results_.update({'test_error': [test_error,
#                                                      test_error_inv]})
#         self.grid.cv_results_.update({'y_pred': inv(y_pred)})
#         self.grid.cv_results_.update({'y_test': inv(self.frame.y_test)})
#         self.grid.cv_results_.update({'X_test': self.frame.X_test})
# 
#         # Calculate the permutation importances
#         perm = permutation_importance(
#                 self.grid.best_estimator_, self.frame.X_test,
#                 self.frame.y_test, scoring='neg_mean_absolute_error',
#                 n_repeats=self.perm_repeats, n_jobs=self.n_jobs,
#                 random_state=self.seed)
#         self.grid.cv_results_.update({'permutations': perm})
# 
#         if print_results:
#             i = self.grid.best_index_
#             cv_results = self.grid.cv_results_
#             print('Best model: ', self.grid.best_estimator_)
#             print('CV mean score: ', -cv_results['mean_test_score'][i])
#             print('CV std score: ', cv_results['std_test_score'][i])
#             print('Test error = {}'.format(test_error))
#             print('Test inverted error = {}'.format(test_error_inv))
# 
#     def dump_results(self):
#         """Saves the sklearn grid."""
#         labels = ""
#         for i, feature in enumerate(self.frame.labels):
#             if i == 0:
#                 labels += feature
#             elif i == len(self.frame.labels) - 1:
#                 labels += "_" + feature
#             else:
#                 labels += "_" + feature
#         fpath = "../data/fits/fit_{}_{}@{}@_@{}@.p".format(
#                 self.name, self.frame.subhalos, self.frame.target, labels)
#         print('Saving the grid search to {}'.format(fpath))
#         dump(self.grid, fpath)
# 
#     def evaluate(self):
#         """A shortcut to evaluate the model."""
#         self.fit_grid()
#         self.score_model()
#         self.dump_results()
