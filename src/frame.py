import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit


class Frame:

    def __init__(self, features, target, subhalos, split='stratify',
                 test_size=0.2, dy=0.2, seed=42):
        self.features_pipe = None
        self.target_pipe = None
        self.labels = [p for p in features.keys()]
        self.target = target['label']
        self.subhalos = subhalos

        X, y = self.get_raw_data(features, target, subhalos)
        X_train, X_test, y_train, y_test = self.split_data(
                X, y, split, test_size, dy, seed)

        # Apply transformations
        self.fit_scaler(X_train, y_train, features, target)
        self.X_train = self.features_pipe.transform(X_train)
        self.X_test = self.features_pipe.transform(X_test)
        self.y_train = self.target_pipe.transform(y_train).reshape(-1,)
        self.y_test = self.target_pipe.transform(y_test).reshape(-1,)

    def get_raw_data(self, features, target, subhalos):
        """Loads the raw data and applies log10 transform if any specified."""
        # Loads the matched data
        match = np.load('/mnt/zfsusers/rstiskalek/hydro/data/matched.npy')
        if subhalos not in [0, 1, 2]:
            raise ValueError("Invalid handle: 'subhalos': {}".format(subhalos))
        if subhalos == 0:
            mask = np.ones_like(match['level'], dtype=bool)
        else:
            mask = match['level'] == subhalos

        # Get the target
        y = match[mask][target['label']].reshape(-1, 1)
        if 'log10' in target['transforms']:
            y = np.log10(y)

        # Get the features
        X = [None] * len(features)
        for i, (key, value) in enumerate(features.items()):
            X[i] = match[mask][key]
            if 'log10' in value:
                X[i] = np.log10(X[i])
        X = np.vstack(X).T
        return X, y

    def split_data(self, X, y, split, test_size, dy, seed):
        """Splits the data."""
        y_sorted = np.sort(y.reshape(-1))
        N = len(y_sorted)
        ymin = y_sorted[int(0.01 * N)]
        ymax = y_sorted[int(0.99 * N)]
        bins = np.arange(ymin, ymax, dy)

        bands = np.digitize(y, bins)

        sss = StratifiedShuffleSplit(n_splits=50, test_size=test_size,
                                     random_state=seed)
        for train_index, test_index in sss.split(X, bands):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        return X_train, X_test, y_train, y_test

    def fit_scaler(self, X_train, y_train, features, target):
        """Pass"""
        cols_standard = []
        cols_minmax = []
        for i, (key, value) in enumerate(features.items()):
            if 'standard' in value:
                cols_standard.append(i)
            elif 'minmax' in value:
                cols_minmax.append(i)

        self.features_pipe = ColumnTransformer(
                [('standard', StandardScaler(), cols_standard),
                 ('minmax', MinMaxScaler(), cols_minmax)],
                remainder='passthrough')

        if 'standard' in target['transforms']:
            self.target_pipe = StandardScaler()
        else:
            raise ValueError('Unsupported target transform.')

        self.features_pipe.fit_transform(X_train)
        self.target_pipe.fit_transform(y_train)
