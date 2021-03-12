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

from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV)
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
from joblib import (dump, parallel_backend)


class RegressionModel:
    """A simple regression model."""

    def __init__(self, model, name, parameter_grid, frame, search='random',
                 n_iters=10, n_jobs=1, verbose=3, seed=42, cv=5,
                 perm_repeats=25):
        self.name = name
        self.n_jobs = n_jobs
        self.frame = frame
        self.seed = seed
        self.perm_repeats = perm_repeats
        self.cv = cv
        # Initialise the grids
        if search == 'random':
            self.grid = RandomizedSearchCV(model, parameter_grid,
                                           n_iter=n_iters,
                                           scoring='neg_mean_absolute_error',
                                           cv=cv, verbose=verbose)
        elif search == 'grid':
            self.grid = GridSearchCV(model, parameter_grid,
                                     scoring='neg_mean_absolute_error', cv=cv,
                                     verbose=verbose)
        else:
            raise ValueError("Invalid search option")

    def fit_grid(self):
        """Fits the model."""
        with parallel_backend('loky', n_jobs=self.n_jobs):
            self.grid.fit(self.frame.X_train, self.frame.y_train)
        # Append the column names
        self.grid.cv_results_.update({'columns': self.frame.labels})
        self.grid.cv_results_.update({'target': self.frame.target})

    def score_model(self, print_results=True):
        """Scores the models on the test data."""
        y_pred = self.grid.best_estimator_.predict(self.frame.X_test)

        # Transform back
        inv = self.frame.target_pipe.inverse_transform

        test_error = mean_absolute_error(self.frame.y_test, y_pred)
        test_error_inv = mean_absolute_error(inv(self.frame.y_test),
                                             inv(y_pred))
        # Append the test batch error and the test data
        self.grid.cv_results_.update({'test_error': [test_error,
                                                     test_error_inv]})
        self.grid.cv_results_.update({'y_pred': inv(y_pred)})
        self.grid.cv_results_.update({'y_test': inv(self.frame.y_test)})
        self.grid.cv_results_.update({'X_test': self.frame.X_test})

        # Calculate the permutation importances
        perm = permutation_importance(
                self.grid.best_estimator_, self.frame.X_test,
                self.frame.y_test, scoring='neg_mean_absolute_error',
                n_repeats=self.perm_repeats, n_jobs=self.n_jobs,
                random_state=self.seed)
        self.grid.cv_results_.update({'permutations': perm})

        if print_results:
            i = self.grid.best_index_
            cv_results = self.grid.cv_results_
            print('Best model: ', self.grid.best_estimator_)
            print('CV mean score: ', -cv_results['mean_test_score'][i])
            print('CV std score: ', cv_results['std_test_score'][i])
            print('Test error = {}'.format(test_error))
            print('Test inverted error = {}'.format(test_error_inv))

    def dump_results(self):
        """Saves the sklearn grid."""
        labels = ""
        for i, feature in enumerate(self.frame.labels):
            if i == 0:
                labels += feature
            elif i == len(self.frame.labels) - 1:
                labels += "_" + feature
            else:
                labels += "_" + feature
        fpath = "../data/fits/fit_{}_{}@{}@_@{}@.p".format(
                self.name, self.frame.subhalos, self.frame.target, labels)
        print('Saving the grid search to {}'.format(fpath))
        dump(self.grid, fpath)

    def evaluate(self):
        """A shortcut to evaluate the model."""
        self.fit_grid()
        self.score_model()
        self.dump_results()
