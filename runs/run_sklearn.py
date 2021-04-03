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
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import toml

# Make galofeats pip installable at some point to avoid this
import sys
sys.path.append('../')

from galofeats import (UnionPipeline, DataFrameSelector, stratify_split,
                       SklearnRegressor)


models = {'ExtraTreesRegressor': ExtraTreesRegressor,
          'MLPRegressor': MLPRegressor,
          'LinearRegression': LinearRegression}

scalers = {'MinMaxScaler': preprocessing.MinMaxScaler,
           'QuantileTransformer': preprocessing.QuantileTransformer,
           'RobustScaler': preprocessing.RobustScaler,
           'StandardScaler': preprocessing.StandardScaler}

class ConfigParser:
    """Add documentantion. Especially since may have to reuse this."""


    def __init__(self, path):
        self.cnf = toml.load(path)

    @property
    def grid(self):
        est_cnf = self.cnf['Estimator']
        estimator = models[est_cnf['model']](**est_cnf['params'])
        return GridSearchCV(estimator, **self.cnf['Grid'])

    def _pipeline(self, kind):
        data = self.cnf['Data'][kind]
        pipe = [None] * len(data)
        for i, (key, value) in enumerate(data.items()):
            print(value)
            scaler = scalers[value['scaler']](**value['kwargs'])
            if value['log']:
                pipe[i] = Pipeline([('selector', DataFrameSelector(key, key)),
                                    ('scaler', scaler)])
            else:
                pipe[i] = Pipeline([('selector', DataFrameSelector(key)),
                                    ('scaler', scaler)])
        return UnionPipeline(pipe)

    @property
    def feature_pipeline(self):
        return self._pipeline('features')

    @property
    def target_pipeline(self):
        return self._pipeline('target')

    @property
    def data(self):
        data = numpy.load(self.cnf['Data']['split']['path'])
        features = self.feature_pipeline.attributes
        target = self.target_pipeline.attributes
        if len(target) != 1:
            raise ValueError("Only a single target supported")
        target = target[0]
        kwargs = self.cnf['Data']['split']['kwargs']
        kwargs.update({'seed': self.cnf['Main']['seed']})
        return stratify_split(data, features, target, **kwargs)


def run_regressor():
    # And decide what to save
    pass
    # Args to specify the config file

    # Fit the model, calculate permutation importance, feature importance,
    # possibly the learning curve and dump the results in a meaningful way.

    # Under what name to save this? Save the config file, scores and grid cv






def main():
    cnf = ConfigParser('config.toml')
    regressor = SklearnRegressor(cnf.grid, *cnf.data, cnf.feature_pipeline,
                                 cnf.target_pipeline)
    regressor.fit()
    print(regressor.score())

    print(regressor)

if __name__ == "__main__":
    main()
