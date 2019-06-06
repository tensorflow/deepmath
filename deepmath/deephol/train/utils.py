"""Utility functions for Holparam code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


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
      init = tf.contrib.lookup.KeyValueTensorInitializer(keys, values)
      return tf.contrib.lookup.HashTable(init, 1)
    else:
      init = tf.contrib.lookup.KeyValueTensorInitializer(values, keys)
      return tf.contrib.lookup.HashTable(init, '')
