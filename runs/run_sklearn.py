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

"""A submission script for sklearn regressors."""

import sys
import os
import argparse

import numpy
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from joblib import dump

from galofeats import (UnionPipeline, DataFrameSelector, stratify_split,
                       SklearnRegressor)
import toml


# Dictionary of models to allow quick initialisation by key
MODELS = {'ExtraTreesRegressor': ExtraTreesRegressor,
          'MLPRegressor': MLPRegressor,
          'LinearRegression': LinearRegression}

# Dictionary of scalers to allow quick initialisation by key
SCALERS = {'MinMaxScaler': preprocessing.MinMaxScaler,
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
        estimator = MODELS[est_cnf['model']](**est_cnf['params'])
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
            scaler = SCALERS[value['scaler']](**kwargs)
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
    """
    A quick and dirty submission script that summarises the model, and the
    fitting results in .txt files and also scatter plots the predicted target
    variables.

    Parameters
    ----------
    regressor : `galofeats.SklearnRegressor` object

    cnf : `ConfigParser` object
    """

    def __init__(self, regressor, cnf):
        self.regressor = regressor
        self.cnf = cnf
        # Enforce correct ending of the file path
        path = self.cnf.cnf['Main']['folder']
        if not path.endswith('/'):
            path += '/'
        self.path = path
        # Setup the output folder
        self.setup_folder()

    def setup_folder(self):
        """
        Prepares the output folder. The folder is created or, if already
        in existence, all '.txt', '.p', and '.png' files are removed.
        """
        if os.path.exists(self.path):
            files = os.listdir(self.path)
            extensions = ['.txt', '.p', '.png']
            for item in files:
                for ext in extensions:
                    if item.endswith(ext):
                        os.remove(self.path + item)
                        continue
        else:
            os.mkdir(self.path)

    def print_model_summary(self):
        """Prints summary of this model - the data and hyperparameters."""
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

        # Data
        print("Data:")
        print("-"*79)
        print(self.cnf.cnf['Data']['path'])

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
        """Parses regressor sub-dictionaries and appends seed and n_jobs."""
        main = self.cnf.cnf['Main']
        kwargs = self.cnf.cnf['Regressor'][kind]
        # Append the seed and n_jobs
        kwargs.update({'seed': main['seed'],
                       'n_jobs': main['n_jobs']})
        return kwargs

    def print_fit_summary(self, scores, perm, feat_imp, learning_curve):
        """Prints results of the fitting."""
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
        ytest = self.regressor.y_test[target]
        ypred = self.regressor.predict()[target]
        try:
            log_target = self.cnf.cnf['Data']['target'][target]['log']
            kind = 'dex'
        except KeyError:
            log_target = False
            kind = ''
        if log_target:
            ypred = numpy.log10(ypred)
            ytest = numpy.log10(ytest)
        dy = numpy.abs(ytest - ypred)

        print("Inverted MAE: {:.4f} +- {:.4f} {}"
              .format(numpy.mean(dy), numpy.std(dy), kind))
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

    def plot_scatter(self):
        """
        Makes a 2-row scatter plot for each feature. The top figure is a
        scatter plot of predicted and true target vs feature, calculated
        on the test set, and the bottom figure shows the residuals.
        """
        X = self.regressor.X_test
        y_pred = self.regressor.predict()
        y_test = self.regressor.y_test

        target = y_test.dtype.names[0]
        try:
            log_target = self.cnf.cnf['Data']['target'][target]['log']
        except KeyError:
            log_target = False
        if log_target:
            y_pred[target] = numpy.log10(y_pred[target])
            y_test[target] = numpy.log10(y_test[target])

        for attr in X.dtype.names:
            try:
                log_target = self.cnf.cnf['Data']['features'][attr]['log']
                if log_target:
                    X[attr] = numpy.log10(X[attr])
            except KeyError:
                log_target = False

            # Make the figure
            fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
            fig.subplots_adjust(hspace=0)
            axes[0].scatter(X[attr], y_test[target], s=3, label='Test set')
            axes[0].scatter(X[attr], y_pred[target], s=3, label='Predicted',
                            alpha=0.5)
            axes[0].legend()

            axes[1].scatter(X[attr], y_pred[target] - y_test[target], s=3)
            if log_target:
                axes[1].set_ylabel('Residuals (dex)')
            else:
                axes[1].set_ylabel('Residuals ({})'.format(attr))
            axes[1].axhline(0, c='orange', ls='--')

            axes[0].set_ylabel(target)
            axes[1].set_xlabel(attr)
            fig.savefig(self.path + '{}.png'.format(attr))
            plt.close(fig)

    def run(self):
        """
        One button to run them all. The summarises the models, fits it,
        and inspects it and generates the appropriate summarising files.
        """
        # Save the system buffer
        stdout0 = sys.stdout
        # Print model summary to a txt file
        with open(self.path + 'parameters.txt', 'w') as f:
            sys.stdout = f
            self.print_model_summary()
            sys.stdout = stdout0
        # Do the heavy work
        self.regressor.fit()
        scores = self.regressor.score()
        # Permutation importance
        kwargs0 = self.importance_kwargs('permutation_importance')
        perm = self.regressor.permutation_importance(**kwargs0)
        # Feature importance for ensemble models
        kwargs1 = self.importance_kwargs('feature_importance')
        feat_imp = self.regressor.feature_importance(**kwargs1)
        # Learning curve (optional)
        kwargs2 = self.importance_kwargs('learning_curve')
        make_curve = kwargs2.pop('to_calculate')
        if make_curve:
            learning_curve = self.regressor.learning_curve(**kwargs2)
        else:
            learning_curve = None
        # Print model results to a txt file
        with open(self.path + 'results.txt', 'w') as f:
            sys.stdout = f
            self.print_fit_summary(scores, perm, feat_imp, learning_curve)
            sys.stdout = stdout0

        # Make the scatter plots
        self.plot_scatter()
        # At last, save the fitted grid
        dump(self.regressor.grid, self.path + 'grid.p')


def main():
    plt.switch_backend('agg')
    parser = argparse.ArgumentParser(description='Regressor submitter.')
    parser.add_argument('--path', type=str, help='Config file path.')
    args = parser.parse_args()

    cnf = ConfigParser(args.path)
    regressor = SklearnRegressor(cnf.grid, *cnf.data, cnf.feature_pipeline,
                                 cnf.target_pipeline)
    model = SubmissionModel(regressor, cnf)
    model.run()


if __name__ == "__main__":
    main()
