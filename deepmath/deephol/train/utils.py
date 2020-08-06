"""Utility functions for Holparam code."""
import tensorflow.compat.v1 as tf
from tensorflow.contrib import lookup as contrib_lookup


class Params(dict):
  """Very simple Hyperparameter wrapper around dictionaries."""

  def __init__(self, *args, **kwargs):
    super(Params, self).__init__(*args, **kwargs)
    self.__dict__ = self


def vocab_table_from_file(filename, reverse=False):
  with tf.gfile.Open(filename, 'r') as f:
    keys = [s.strip() for s in f.readlines()]
    values = tf.range(len(keys), dtype=tf.int64)
    if not reverse:
      init = contrib_lookup.KeyValueTensorInitializer(keys, values)
      return contrib_lookup.HashTable(init, 1)
    else:
      init = contrib_lookup.KeyValueTensorInitializer(values, keys)
      return contrib_lookup.HashTable(init, '')
