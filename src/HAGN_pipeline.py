"""Pipeline for ..."""
import numpy as np
from scipy import stats
from argparse import ArgumentParser

from sklearn.model_selection import (StratifiedShuffleSplit, train_test_split,
                                     cross_val_predict, GridSearchCV,
                                     RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (Ridge, HuberRegressor)
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from joblib import dump, parallel_backend

SEED = 42
CV = 10

parser = ArgumentParser(description='Pipeline to test models defined at the\
                                     end of this file.')
parser.add_argument('--n_jobs', type=int, default=1, help='number of jobs')
parser.add_argument('--n_iters', type=int, default=10,
                    help='number of iterations')
parser.add_argument('--subhalos', type=int, default=1,
                    help='whether to include subhalos')
args = parser.parse_args()


class Frame:
    """Only include inputs that I will want to change on the go."""

    def __init__(self, subhalos=0, data_fraction=1, seed=SEED,
                 split='stratify'):
        self._subhalos = None
        self._data_fraction = None
        self._split = None

        self.subhalos = subhalos
        self.data_fraction = data_fraction
        self.seed = SEED
        self.split = split
        data = self.load_data()
        print('Initial size ', data.size)
        # Possibly subsample
        data = self.subsample_data(data)
        print('Final size ', data.size)
        # Stratify
        X_train, X_test = self.split_data(data)
        # Split labels/features
        X_train, Y_train = self.split_labels_features(X_train)
        X_test, Y_test = self.split_labels_features(X_test)
        self.columns = X_train.dtype.names
        # Switch to unstructered array
        self.X_train = self.structured2unstructured(X_train)
        self.X_test = self.structured2unstructured(X_test)
        self.Y_train = self.structured2unstructured(Y_train).reshape(-1,)
        self.Y_test = self.structured2unstructured(Y_test).reshape(-1,)
        # Scaling
        self.X_train = self.scale_data(self.X_train)
        self.X_test = self.scale_data(self.X_test)

    def load_data(self):
        """Loads the raw matched data begins the formatting."""
        data = np.load('/mnt/zfsusers/rstiskalek/hydro/data/matched.npy')
        if self.subhalos != 0:
            data = data[data['level'] == self.subhalos]
        data = self.remove_fields(data, ['x', 'y', 'z', 'level', 'cvel',
                                         'concentration'])
#        data = remove_fields(data, ['x', 'y', 'z', 'level', 'r', 'rvir',
#                                    'spin', 'L', 'cvel', 'concentration'])

        # Taking log10 of these helps a lot
        pars = ['Mvir', 'MS', 'r', 'rvir', 'rho0', 'spin', 'L',
                'concentration']
        for p in pars:
            if p not in data.dtype.names:
                print('{} not included, thus not being log10(ed)'.format(p))
                continue
            data[p] = np.log10(data[p])
        return data

    def subsample_data(self, data):
        """
        Subsamples the data so that only a fraction of it is used to train and
        test the model.
        """
        if self.data_fraction == 1:
            return data
        if self.split == 'random':
            X, __ = train_test_split(data, test_size=(1-self.data_fraction),
                                     random_state=self.seed * 2)
            return X
        band_width = 0.4  # dex logMvir
        bands = np.floor((data['Mvir'] - data['Mvir'].min()) / band_width)
        split = StratifiedShuffleSplit(n_splits=20,
                                       test_size=(1-self.data_fraction),
                                       random_state=self.seed * 2)
        for train_index, __ in split.split(data, bands):
            X = data[train_index]
        return X

    def split_data(self, data):
        """Splits the data following either the random or stratify approach."""
        if self.split == 'random':
            X_train, X_test = train_test_split(data, test_size=0.2,
                                               random_state=self.seed)
            return X_train, X_test
        test_ratio = 0.2
        band_width = 0.4  # dex logMvir
        bands = np.floor((data['Mvir'] - data['Mvir'].min()) / band_width)
        split = StratifiedShuffleSplit(n_splits=20, test_size=test_ratio,
                                       random_state=self.seed)
        for train_index, test_index in split.split(data, bands):
            X_train = data[train_index]
            X_test = data[test_index]
        return X_train, X_test

    def scale_data(self, X):
        """
        Scales the features. For now applies the standard scaler to
        everything.
        """
        scaler = StandardScaler()
        return scaler.fit_transform(X)

    def split_labels_features(self, X):
        """Splits the target feature from the vector of labels."""
        X_labels = list(X.dtype.names)

        Y_labels = ['MS']  # Only one label supported
        X_labels_remove = X_labels.copy()
        for label in Y_labels:
            X_labels_remove.remove(label)

        Y = self.remove_fields(X, X_labels_remove)
        X = self.remove_fields(X, Y_labels)
        return X, Y

    @staticmethod
    def remove_fields(arr, labels):
        """Removes fields ``labels`` from a structured array ``arr``."""
        names = list(arr.dtype.names)
        for name in arr.dtype.names:
            if name in labels:
                names.remove(name)

        dtype = {'names': names, 'formats': ['float64'] * len(names)}
        out = np.zeros(arr.size, dtype=dtype)
        for name in names:
            out[name] = arr[name]
        return out

    @staticmethod
    def structured2unstructured(arr):
        """
        Transforms a structured array into one of shape (Nsamples, Nfeatures).
        """
        names = arr.dtype.names
        out = np.zeros(shape=(arr.size, len(names)))
        for i, name in enumerate(names):
            out[:, i] = arr[name]
        return out

    @property
    def subhalos(self):
        """
        Whether to include subhalos.

        Options:
            - 0: include both halos and subhalos,
            - 1: include only halos,
            - 2: include only subhalos.
        """
        return self._subhalos

    @subhalos.setter
    def subhalos(self, subhalos):
        if subhalos not in [0, 1, 2]:
            raise ValueError("Invalid input. Available options: 0, 1, 2.")
        self._subhalos = subhalos

    @property
    def data_fraction(self):
        """Fraction of the data set to be used to train and test the model."""
        return self._data_fraction

    @data_fraction.setter
    def data_fraction(self, fraction):
        if not 0 < fraction <= 1.:
            raise ValueError("``fraction`` must be (0, 1)")
        self._data_fraction = fraction

    @property
    def split(self):
        """Whether to do a random or stratified split."""
        return self._split

    @split.setter
    def split(self, split):
        if split not in ['random', 'stratify']:
            raise ValueError("Allowed choices: 'random' and 'stratify'")
        self._split = split

    def __call__(self):
        """Returns the split and scaled data."""
        return self.X_train, self.Y_train, self.X_test, self.Y_test


def test_model(model, params, name, subhalos, search='random', n_iters=10,
               n_jobs=1, verbose=3):
    """Tests a specific model over a parameter grid using a specific seach."""
    frame = Frame(subhalos=subhalos, data_fraction=1, split='stratify')
    X_train, Y_train, X_test, Y_test = frame()
    print('Running model {}:{} with {} jobs for {} iterations'
          .format(name, subhalos, n_jobs, n_iters))
    if search == 'random':
        grid = RandomizedSearchCV(model, params, cv=CV,
                                  scoring='neg_mean_absolute_error',
                                  verbose=verbose, n_iter=n_iters,
                                  pre_dispatch='2*n_jobs')
    elif search == 'grid':
        grid = GridSearchCV(model, params, cv=CV,
                            scoring='neg_mean_absolute_error',
                            verbose=verbose)
    else:
        raise ValueError("Invalid search option")

    with parallel_backend('loky', n_jobs=n_jobs):
        grid.fit(X_train, Y_train)
    print('Best model: ', grid.best_estimator_)
    # Append the column names
    grid.cv_results_.update({'columns': frame.columns})

    i = grid.best_index_
    cv_results = grid.cv_results_
    print('Best model mean score: ', -cv_results['mean_test_score'][i])
    print('Best model std score: ', cv_results['std_test_score'][i])

    best_model = grid.best_estimator_
    Y_test_predicted = cross_val_predict(best_model, X_test, Y_test, cv=CV)
    test_error = mean_absolute_error(Y_test, Y_test_predicted)
    print('Best model test error = {}'.format(test_error))

    # Append the test batch error
    grid.cv_results_.update({'test_error': test_error})

    fpath = "../data/fit_{}_subhalos_{}.p".format(name, subhalos)
    print('Saving the grid search to {}'.format(fpath))
    dump(grid, fpath)

    if name == 'ExtraTreesRegressor':
        print("Calculating feature importances using the best model.")
        seeds = np.random.randint(0, int(2**32), size=100)
        feature_importances = [None] * len(seeds)
        kwargs = grid.best_params_
        for i, seed in enumerate(seeds):
            kwargs.update({'random_state': seed})
            model = ExtraTreesRegressor(**kwargs)
            model.fit(X_train, Y_train)
            feature_importances[i] = model.feature_importances_

        feature_importances = np.vstack(feature_importances).T
        fpath = "../data/feat_importances_{}_subhalos_{}.p".format(name,
                                                                   subhalos)
        print("Saving feature importances to {}".format(fpath))
        dump(feature_importances, fpath)


models = {'Ridge': Ridge(alpha=1.0, tol=1e-5),
          'HuberRegressor': HuberRegressor(max_iter=10000),
          'RandomForestRegressor': RandomForestRegressor(),
          'ExtraTreesRegressor': ExtraTreesRegressor(),
          'MLPRegressor': MLPRegressor()}

# Note that for random forests the 'mae' criterion is very slow
params = {'Ridge': {'alpha': stats.uniform(loc=0, scale=10)},
          'HuberRegressor': {'epsilon': stats.uniform(scale=1, loc=5),
                             'alpha': stats.uniform(scale=0, loc=2.5),
                             'warm_start': [True, False]},
          'RandomForestRegressor': {'n_estimators': [2, 4, 8, 16, 32, 64, 128,
                                                     256, 512],
                                    'criterion': ['mse'],
                                    'max_depth': [None, 16, 32],
                                    'min_samples_split': [2, 4, 8, 16, 32],
                                    'min_samples_leaf': [2, 4, 8, 16, 32],
                                    'max_leaf_nodes': [None, 32, 64, 128],
                                    'max_features': ['auto', 'log2'],
                                    'min_impurity_decrease': [0.0, 0.1, 0.25],
                                    'warm_start': [True]},
          'ExtraTreesRegressor': {'n_estimators': [2, 4, 8, 16, 32, 64, 128,
                                                   256, 512],
                                  'criterion': ['mse'],
                                  'max_depth': [None, 16, 32],
                                  'min_samples_split': [2, 4, 8, 16, 32],
                                  'min_samples_leaf': [2, 4, 8, 16, 32],
                                  'max_leaf_nodes': [None, 32, 64, 128],
                                  'max_features': ['auto', 'log2'],
                                  'min_impurity_decrease': [0.0, 0.1, 0.25],
                                  'warm_start': [True]},
          'MLPRegressor': {'activation': ['relu', 'logistic', 'tanh'],
                           'solver': ['adam', 'sgd'],
                           'alpha': [0.0001, 0.001, 0.1, 1],
                           'learning_schedule': ['adaptive'],
                           'learning_rate': [0.001, 0.01],
                           'max_iter': [1000],
                           'warm_start': [True],
                           'early_stooping': [True, False]}}

# for name in ['Ridge', 'HuberRegressor']:
# for name in ['RandomForestRegressor', 'ExtraTreesRegressor']:
for name in ['ExtraTreesRegressor']:
    test_model(models[name], params[name], name, subhalos=args.subhalos,
               search='random', n_iters=args.n_iters, n_jobs=args.n_jobs,
               verbose=1)
