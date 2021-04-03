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

"""Submission script."""
import sys

import numpy
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import toml

# Make galofeats pip installable at some point to avoid this
sys.path.append('../')

from galofeats import (UnionPipeline, DataFrameSelector, stratify_split,
                       SklearnRegressor)

# Dictionary of models to allow quick initialisation by key
models = {'ExtraTreesRegressor': ExtraTreesRegressor,
          'MLPRegressor': MLPRegressor,
          'LinearRegression': LinearRegression}

# Dictionary of scalers to allow quick initialisation by key
scalers = {'MinMaxScaler': preprocessing.MinMaxScaler,
           'QuantileTransformer': preprocessing.QuantileTransformer,
           'RobustScaler': preprocessing.RobustScaler,
           'StandardScaler': preprocessing.StandardScaler}

class ConfigParser:
    """
    Config parser tailored to `galofeats.SklearnRegressor` hyperparameter
    grid search.

    Notes:
        - Add support for sklearn.model_selection.RandomizedSearchCV
        - Summarise what was initialised

    Parameters
    ----------
    path : str
        Filepath to the toml config file.

    """

    def __init__(self, path):
        self.cnf = toml.load(path)

    @property
    def grid(self):
        """
        Parses the config file and initialises the hyperparameter grid.

        Returns
        -------
        grid : sklearn hyperparameter grid object
        """
        est_cnf = self.cnf['Estimator']
        estimator = models[est_cnf['model']](**est_cnf['params'])
        kwargs = self.cnf['Grid']
        kwargs.update({'n_jobs': self.cnf['Main']['n_jobs']})
        return GridSearchCV(estimator, **kwargs)

    def _pipeline(self, kind):
        """
        A support function to initialise either the feature or target
        preprocessing pipeline.

        Parameters
        ----------
        kind : str
            Either `features` or `target`.

        Returns
        -------
        pipeline : `galofeats.UnionPipeline`
            An initialised pipeline.
        """
        data = self.cnf['Data'][kind]
        pipe = [None] * len(data)
        for i, (key, value) in enumerate(data.items()):
            # get scaler kwargs
            try:
                kwargs = value['kwargs']
            except KeyError:
                kwargs = {}
            scaler = scalers[value['scaler']](**kwargs)
            # whether to logarithm transform this value
            try:
                to_log = value['log']
            except KeyError:
                to_log = False
            if to_log:
                pipe[i] = Pipeline([('selector', DataFrameSelector(key, key)),
                                    ('scaler', scaler)])
            else:
                pipe[i] = Pipeline([('selector', DataFrameSelector(key)),
                                    ('scaler', scaler)])
        return UnionPipeline(pipe)

    @property
    def feature_pipeline(self):
        """Feature preprocessing pipeline (`galofeats.UnionPipeline`)"""
        return self._pipeline('features')

    @property
    def target_pipeline(self):
        """Target preprocessing pipeline (`galofeats.UnionPipeline`)"""
        return self._pipeline('target')

    @property
    def data(self):
        """
        Loaded training and test set and performs a stratified
        split.

        Returns
        -------
        X_train : numpy.ndarray with named fields
            Train features.
        X_test : numpy.ndarray with named fields
            Test features.
        y_train : numpy.ndarray with named fields
            Train target.
        y_test : numpy.ndarray with named fields
            Test target.
        """
        data = numpy.load(self.cnf['Data']['path'])
        features = self.feature_pipeline.attributes
        target = self.target_pipeline.attributes
        if len(target) != 1:
            raise ValueError("Only a single target supported")
        target = target[0]
        kwargs = self.cnf['Data']['split']['kwargs']
        kwargs.update({'seed': self.cnf['Main']['seed']})
        return stratify_split(data, features, target, **kwargs)


class SubmissionModel:

    def __init__(self, regressor, cnf):
        self.regressor = regressor
        self.cnf = cnf

    def print_model_summary(self):
        print("Grid:")
        print("-"*79)
        print(self.regressor.grid)
        print()

        print("Stratify split kwargs:")
        print("-"*79)
        kwargs = self.cnf.cnf['Data']['split']['kwargs']
        kwargs.update({'seed': self.cnf.cnf['Main']['seed']})
        for key, value in kwargs.items():
            print('{}: {}'.format(key, value))
        print()

        # Features summary
        print("Features:")
        print("-"*79)
        for pipe in self.regressor.feature_pipeline.pipelines:
            selector = pipe.steps[0][1]
            print("Attributes: {}, log attributes: {}"
                  .format(selector.attributes, selector.log_attributes))
            print("Scaler: {}".format(pipe.steps[1][1]))
            print()

        # Target summary
        print("Target:")
        print("-"*79)
        for pipe in self.regressor.target_pipeline.pipelines:
            selector = pipe.steps[0][1]
            print("Attributes: {}, log attributes: {}"
                  .format(selector.attributes, selector.log_attributes))
            print("Scaler: {}".format(pipe.steps[1][1]))
            print()

        # Permutation importance
        print("Permutation importance:")
        print("-"*79)
        for key, value in (self.importance_kwargs('permutation_importance')
                           .items()):
            print('{}: {}'.format(key, value))
        print()

        # Permutation importance
        print("Feature importance:")
        print("-"*79)
        for key, value in (self.importance_kwargs('feature_importance')
                           .items()):
            print('{}: {}'.format(key, value))
        print()

        # Learning curve
        print("Learning curve:")
        print("-"*79)
        for key, value in (self.importance_kwargs('learning_curve')
                           .items()):
            print('{}: {}'.format(key, value))
        print()

    def importance_kwargs(self, kind):
        main = self.cnf.cnf['Main']
        kwargs = self.cnf.cnf['Regressor'][kind]

        kwargs.update({'seed': main['seed'],
                       'n_jobs': main['n_jobs']})
        return kwargs

    def print_fit_summary(self, scores, perm, feat_imp, learning_curve):
        # Best model
        print("Best estimator:")
        print("-"*79)
        print(self.regressor.model)
        print()

        # Scores
        print("Best estimator scores:")
        print("-"*79)
        for key, value in scores.items():
            print('{}: {:.4f}'.format(key, value))
        # Get the real inverted score and return dex
        target = self.regressor.target_pipeline.attributes[0]
        ytest = numpy.log10(self.regressor.y_test[target])
        ypred = numpy.log10(self.regressor.predict()[target])
        dy = numpy.abs(ytest - ypred)
        print("Inverted MAE: {:.4f} +- {:.4f} dex".format(numpy.mean(dy),
                                                          numpy.std(dy)))
        print()

        features = self.regressor.feature_pipeline.attributes
        # Permutation importance
        print("Permutation importance scores:")
        print("-"*79)
        for i in numpy.argsort(perm[:, 0])[::-1]:
            print('{}: {:.4f} +- {:.4f}'.format(features[i], perm[i, 0],
                                                perm[i, 1]))
        print()

        # Feature importance
        if feat_imp is not None:
            print("Feature importance scores:")
            print("-"*79)
            for i in numpy.argsort(feat_imp[:, 0])[::-1]:
                print("{}: {:.4f} +- {:.4f}"
                      .format(features[i], feat_imp[i, 0],
                              feat_imp[i, 1]))
            print()
        # Learning curve 
        if learning_curve is not None:
            print("Learning curve:")
            print("-"*79)
            test_score = numpy.mean(learning_curve['test_scores'], axis=1)
            for i, score in enumerate(test_score):
                print("Size: {} -> {:.4f}"
                      .format(learning_curve['train_sizes'][i], score))

            print()

    def run(self):
        stdout0= sys.stdout
        # Print model summary to a txt file
        with open('parameters.txt', 'w') as f:
            sys.stdout = f
            self.print_model_summary()
            sys.stdout = stdout0

        self.regressor.fit()

        scores = self.regressor.score()
        perm_kwargs = self.importance_kwargs('permutation_importance')
        perm = self.regressor.permutation_importance(**perm_kwargs)

        feat_imp_kwargs = self.importance_kwargs('feature_importance')
        feat_imp = self.regressor.feature_importance(**feat_imp_kwargs)

        learning_curve_kwargs = self.importance_kwargs('learning_curve')
        make_curve = learning_curve_kwargs.pop('to_calculate')
        if make_curve:
            learning_curve = self.regressor.learning_curve(
                    **learning_curve_kwargs)
        else:
            learning_curve = None

        # Print model results to a txt file
        with open('results.txt', 'w') as f:
            sys.stdout = f
            self.print_fit_summary(scores, perm, feat_imp, learning_curve)
            sys.stdout = stdout0

# Add a specific name to the runs
# Now create a folder where to save the files
# Make some plots


def main():
    cnf = ConfigParser('config.toml')
    regressor = SklearnRegressor(cnf.grid, *cnf.data, cnf.feature_pipeline,
                                 cnf.target_pipeline)
    model = SubmissionModel(regressor, cnf)

    model.run()


if __name__ == "__main__":
    main()
