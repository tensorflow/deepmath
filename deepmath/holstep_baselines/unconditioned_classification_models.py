# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Models for classifying proof steps without conditioning on the conjecture."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import layers
from keras.models import Model


def cnn_2x(voc_size, max_len, dropout=0.5):
  """One branch embedding a statement. Binary classifier on top.

  Args:
    voc_size: size of the vocabulary for the input statements.
    max_len: maximum length for the input statements.
    dropout: Fraction of units to drop.
  Returns:
    A Keras model instance.
  """
  statement_input = layers.Input(shape=(max_len,), dtype='int32')

  x = layers.Embedding(
      output_dim=256,
      input_dim=voc_size,
      input_length=max_len)(statement_input)
  x = layers.Convolution1D(256, 7, activation='relu')(x)
  x = layers.MaxPooling1D(3)(x)
  x = layers.Convolution1D(256, 7, activation='relu')(x)
  embedded_statement = layers.GlobalMaxPooling1D()(x)

  x = layers.Dense(256, activation='relu')(embedded_statement)
  x = layers.Dropout(dropout)(x)
  prediction = layers.Dense(1, activation='sigmoid')(x)

  model = Model(statement_input, prediction)
  return model


def cnn_2x_lstm(voc_size, max_len, dropout=0.5):
  """One branch embedding a statement. Binary classifier on top.

  Args:
    voc_size: size of the vocabulary for the input statements.
    max_len: maximum length for the input statements.
    dropout: Fraction of units to drop.
  Returns:
    A Keras model instance.
  """
  statement_input = layers.Input(shape=(max_len,), dtype='int32')
  x = layers.Embedding(
      output_dim=256,
      input_dim=voc_size,
      input_length=max_len)(statement_input)
  x = layers.Convolution1D(256, 7, activation='relu')(x)
  x = layers.MaxPooling1D(3)(x)
  x = layers.Convolution1D(256, 7, activation='relu')(x)
  x = layers.MaxPooling1D(5)(x)
  embedded_statement = layers.LSTM(256)(x)

  x = layers.Dense(256, activation='relu')(embedded_statement)
  x = layers.Dropout(dropout)(x)
  prediction = layers.Dense(1, activation='sigmoid')(x)

  model = Model(statement_input, prediction)
  return model


def embedding_logreg(voc_size, max_len, dropout=0.5):
  """One branch embedding a statement. Binary classifier on top.

  Args:
    voc_size: size of the vocabulary for the input statements.
    max_len: maximum length for the input statements.
    dropout: Fraction of units to drop.
  Returns:
    A Keras model instance.
  """
  statement_input = layers.Input(shape=(max_len,), dtype='int32')

  x = layers.Embedding(
      output_dim=256,
      input_dim=voc_size,
      input_length=max_len)(statement_input)
  x = layers.Flatten()(x)
  x = layers.Activation('relu')(x)
  x = layers.Dropout(dropout)(x)
  prediction = layers.Dense(1, activation='sigmoid')(x)

  model = Model(statement_input, prediction)
  return model


# Contains both the model defition function and the type of encoding needed.
MODELS = {
    'cnn_2x': (cnn_2x, 'integer'),
    'cnn_2x_lstm': (cnn_2x_lstm, 'integer'),
    'embedding_logreg': (embedding_logreg, 'integer'),
}
