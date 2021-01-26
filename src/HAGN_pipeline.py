"""Pipeline for ..."""
import numpy as np

from sklearn.model_selection import (StratifiedShuffleSplit, cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

SEED = 2021


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


def structured2unstructured(arr):
    """Transforms a structured array into one of shape (Npoints, Nfeatures)."""
    names = arr.dtype.names
    out = np.zeros(shape=(arr.size, len(names)))
    for i, name in enumerate(names):
        out[:, i] = arr[name]
    return out


# Throw away some data like position etc
data = np.load('/mnt/zfsusers/rstiskalek/hydro/data/HAGN_matched_catalog.npy')
data = remove_fields(data, ['x', 'y', 'z', 'rs', 'rho0', 'Reff'])

# Split the data
test_ratio = 0.2
band_width = 0.4  # dex logMvir
nsplits = 20
bands = np.floor((data['logMvir'] - data['logMvir'].min()) / band_width)

split = StratifiedShuffleSplit(n_splits=nsplits, test_size=test_ratio,
                               random_state=SEED)
for train_index, test_index in split.split(data, bands):
    X_train = data[train_index]
    X_test = data[test_index]

# Remove the target labels from features
X_labels = list(X_train.dtype.names)
Y_labels = ['logMS']
X_labels_remove = X_labels.copy()
for label in Y_labels:
    X_labels_remove.remove(label)

Y_train = remove_fields(X_train, X_labels_remove)
X_train = remove_fields(X_train, Y_labels)
Y_test = remove_fields(X_test, X_labels_remove)
X_test = remove_fields(X_test, Y_labels)

# Switch to unstructered array
columns = X_train.dtype.names
X_train = structured2unstructured(X_train)
X_test = structured2unstructured(X_test)
Y_train = structured2unstructured(Y_train)
Y_test = structured2unstructured(Y_test)

# Scale features. For now rescale everything using the standard scaler.
# No need for an imputer - all halos have the appropriate labels
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# Linear model
model = LinearRegression()
model.fit(X_train, Y_train)


# Evaluate the model on the training data
train_error = mean_absolute_error(Y_train, model.predict(X_train))
cv = 10
scores = cross_val_score(model, X_train, Y_train,
                         scoring='neg_mean_absolute_error', cv=cv)
scores *= -1  # turn the negative error positive

# Evaluate the model on the test data
test_error = mean_absolute_error(Y_test, model.predict(X_test))

print('Linear model')
print('Absolute errors:')
print('Train error = {}'.format(train_error))
print('CV mean error = {}, CV std error = {}'.format(np.mean(scores),
                                                     np.std(scores)))
print('Test error = {}'.format(test_error))

# • Test different ML algorithms.
# • Think about regularization
# • Plot learning curves - when do you stop? -- gradient methods.
# • Determine the hyperparameters with the validation set
# • What is the final score on the test set for each algorithm?
