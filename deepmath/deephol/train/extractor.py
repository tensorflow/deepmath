"""Extractor for HOLparam models. Tokenizes goals and theorems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from deepmath.deephol.train import utils


class Extractor(object):
  """Extract terms/thms and tokenize based on vocab.

  Attributes:
    params: Hyperparameters.
    goal_table: Lookup table for goal vocab embeddings.
    thms_table: Lookup table for theorem parameter vocab embeddings.
    add_negative: Integer multiple ratio of negative examples to positives.
    all_thms: List of all training thms as strings.
    random_negatives: A batch of negative thm examples.
    goal_closed_negative_iterator: A iterator for negative goal closed examples.
  """

  def __init__(self, params):
    """Inits Extractor class with hyperparameters."""
    self.params = params

    # Create vocab lookup tables from existing vocab id lists.
    dataset_dir = params['dataset_dir']
    goal_file = os.path.join(dataset_dir, params['goal_vocab'])
    self.goal_table = utils.vocab_table_from_file(goal_file)
    if params['thm_vocab'] is not None:
      thms_file = os.path.join(dataset_dir, params['thm_vocab'])
      self.thms_table = utils.vocab_table_from_file(thms_file)
    else:
      self.thms_table = self.goal_table

    # If adding negative examples, create a list of all training thms.
    self.add_negative = params['ratio_neg_examples']
    if self.add_negative:
      # Path to negative examples text file.
      # File should contain one example per line, in the same format as the
      # training examples.
      all_thms_file = os.path.join(dataset_dir, 'thms_ls.train')
      # Get a constant batch_size tensor of tokenized random train theorems.
      d = tf.data.TextLineDataset(all_thms_file)
      d = d.repeat()
      # Shuffle within a sliding window slightly larger than the set of thms.
      d = d.shuffle(
          buffer_size=params.negative_example_shuffle_buffer,
          reshuffle_each_iteration=True)
      d = d.batch(
          (self.params['ratio_neg_examples']) * self.params['batch_size'])
      d = d.make_one_shot_iterator()
      self.random_negatives = d.get_next()

  def tokenize(self, tm, table):
    """Tokenizes tensor string according to lookup table."""
    tm = tf.strings.join(['<START> ', tf.strings.strip(tm), ' <END>'])
    # Remove parentheses - they can be recovered for S-expressions.
    tm = tf.strings.regex_replace(tm, r'\(', ' ')
    tm = tf.strings.regex_replace(tm, r'\)', ' ')
    words = tf.strings.split(tm)
    # Truncate long terms.
    words = tf.sparse.slice(words, [0, 0],
                            [tf.shape(words)[0], self.params.truncate_size])

    word_values = words.values
    id_values = tf.to_int32(table.lookup(word_values))
    ids = tf.SparseTensor(words.indices, id_values, words.dense_shape)
    ids = tf.sparse_tensor_to_dense(ids)
    return ids

  def get_extractor(self):
    """Returns extractor function based on initialized params."""

    def extractor(features, labels):
      """Converts 'goal' features and 'thms' labels to list of ids by vocab."""

      if 'goal' not in features:
        raise ValueError('goal feature missing.')
      if 'tac_id' not in labels:
        raise ValueError('tac_id label missing.')

      # Tile the related features/labels (goals are tiled after embedding).
      goal_tiling_size = self.params.ratio_neg_examples + 1
      labels['tac_id'] = tf.tile(labels['tac_id'], [goal_tiling_size])
      labels['tac_present'] = tf.ones(
          [goal_tiling_size * self.params.batch_size])

      # Tokenize the thm parameter (assumes single thm in list)
      if 'thms' in features:
        if self.add_negative:
          hard_negatives = features['thms_hard_negatives']
          hard_negatives = tf.reshape(tf.transpose(hard_negatives), [-1])

          def hard_or_random_picker(hard_random_pair):
            hard, random = hard_random_pair
            hard_not_present = tf.equal(hard, tf.constant('<NULL>'))
            return tf.cond(hard_not_present, lambda: random, lambda: hard)

          neg_thms = tf.map_fn(
              hard_or_random_picker, (hard_negatives, self.random_negatives),
              dtype=tf.string)
          labels['thm_label'] = tf.concat([
              tf.ones(tf.shape(features['thms'])[0], dtype=tf.int32),
              tf.zeros(tf.shape(neg_thms)[0], dtype=tf.int32)
          ],
                                          axis=0)
          features['thms'] = tf.concat([features['thms'], neg_thms], axis=0)

      if labels is not None and 'thms' in labels:
        labels['thm_ids'] = self.tokenize(labels['thms'], self.thms_table)
        del labels['thms']

      # tokenize 'goal' and 'thms'.
      tf.add_to_collection('goal_string', features['goal'])
      features['goal_ids'] = self.tokenize(features['goal'], self.goal_table)
      del features['goal']
      if 'thms' in features:
        tf.add_to_collection('thm_string', features['thms'])
        features['thm_ids'] = self.tokenize(features['thms'], self.thms_table)
        del features['thms']
        del features['thms_hard_negatives']

      return features, labels

    return extractor
