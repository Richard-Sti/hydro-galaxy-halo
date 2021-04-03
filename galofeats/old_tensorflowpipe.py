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

"""Pipeline for tensorflow models."""
from tensorflow import keras
from tqdm.keras import TqdmCallback


from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance


class TensorFlowRegressionModel:
    """A simple TensorFlow regression model."""

    def __init__(self, model, frame):
        self.model = model
        self.frame = frame

    def fit(self, epochs, validation_split=0.2, batch_size=128, min_delta=0.0,
            patience=50):
        cb = [TqdmCallback(verbose=1),
              keras.callbacks.EarlyStopping(min_delta=min_delta, patience=50)]
        self.model.fit(self.frame.X_train, self.frame.y_train,
                       validation_split=validation_split,
                       batch_size=batch_size, epochs=epochs,
                       verbose=0, callbacks=cb)

    def loss_summary(self):
        history = self.model.history
        loss_args = [history.epoch, history.history['loss']]
        loss_kwargs = {'label': 'Test loss'}
        val_loss_args = [history.epoch, history.history['val_loss']]
        val_loss_kwargs = {'label': 'Validation loss'}
        return loss_args, loss_kwargs, val_loss_args, val_loss_kwargs

    def save_model(self, fpath):
        self.model.save(fpath)

    @classmethod
    def from_saved(cls, fpath, frame):
        model = keras.models.load_model(fpath)
        return cls(model, frame)

    def score_model(self, invert=False):
        """Scores the models on the test data."""
        if not invert:
            return self.model.evaluate(self.frame.X_test, self.frame.y_test)

        inv = self.frame.target_pipe.inverse_transform
        y_pred = self.model.predict(self.frame.X_test)

        error = mean_absolute_error(self.frame.y_test, y_pred)
        error_inv = mean_absolute_error(inv(self.frame.y_test), inv(y_pred))
        return error, error_inv

    def permutation_importance(self, repeats, seed=42):
        perm = permutation_importance(
                self.model, self.frame.X_test,
                self.frame.y_test, scoring='neg_mean_absolute_error',
                n_repeats=repeats, random_state=seed)
        return perm

    def predict_test(self, invert=True):
        y_pred = self.model.predict(self.frame.X_test)
        y_test = self.frame.y_test.copy()
        if invert:
            y_pred = self.frame.target_pipe.inverse_transform(y_pred)
            y_test = self.frame.target_pipe.inverse_transform(y_test)
        return y_test, y_pred
