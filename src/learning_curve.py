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

"""Calculates the learning and validation curves for some model."""

import numpy as np

from frame import Frame

# from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor
# from sklearn.model_selection import learning_curve, validation_curve
from sklearn.model_selection import validation_curve


from joblib import dump

n_jobs = 28


features = {'Mvir': ['log10', 'standard'],
            'concentration': ['log10', 'standard'],
            'rho0': ['log10', 'standard'],
            'gamma': ['standard'],
            'spin': ['log10', 'standard'],
            'potential': ['standard']}

target = {'label': 'Reff', 'transforms': ['log10', 'standard']}

frame = Frame(features, target, subhalos=2, test_size=0.01)

# print('Calculating the learnig curve.')
# # Calculate the learning curve for MLP
# params = {'activation': 'tanh',
#           'alpha': 0.001,
#           'batch_size': 128,
#           'early_stopping': True,
#           'hidden_layer_sizes': (32, 16, 8, 4),
#           'learning_rate': 'adaptive',
#           'solver': 'adam',
#           'max_iter': 5000,
#           'n_iter_no_change': 50,
#           'validation_fraction': 0.2}
# model = MLPRegressor(**params)
#
# train_sizes = np.linspace(0.01, 1.0, 100)
# __, train_score, valid_score, times, __ = learning_curve(
#         model, frame.X_train, frame.y_train, train_sizes=train_sizes, cv=5,
#         scoring='neg_mean_absolute_error', return_times=True, n_jobs=n_jobs)
#
# conv = 1.4475125487928078  # A rough conversion factor to invert the scaling
# learn_res = {'train_sizes': train_sizes,
#              'train_score': train_score / conv,
#              'valid_score': valid_score / conv,
#              'times': times}
#
# dump(learn_res, '../data/learn_curve.p')


print('Calculating the validation curve.')
# Calculate the validation curve for extra trees
params = {'max_depth': 8,
          'max_leaf_nodes': 32,
          'min_samples_split': 128}

model = ExtraTreesRegressor(**params)

prange = np.logspace(np.log10(10), np.log10(1500), 75, dtype=int)
train_score, valid_score = validation_curve(
        model, frame.X_train, frame.y_train, param_name='n_estimators',
        param_range=prange, cv=5, scoring='neg_mean_absolute_error',
        n_jobs=n_jobs)


conv = 1
valid_res = {'parameter_range': prange,
             'train_score': train_score / conv,
             'valid_score': valid_score / conv}

dump(valid_res, '../data/valid_curve.p')
