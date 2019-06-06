"""Provides data for HOL Tactics + Parameters.

Example usage:

mode = tf.estimator.ModeKeys.TRAIN
dataset = data.get_holparam_dataset(mode=mode, dataset_dir=dataset_dir)
input_fn = data.get_input_fn(dataset=dataset, mode=mode, params=params,
                             shuffle_queue=10000,
                             repeat=False)
features, labels = input_fn()
"""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function

import functools
import os
import tensorflow as tf

TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL
PREDICT = tf.estimator.ModeKeys.PREDICT

SOURCE_DATASETDIR = 0
SOURCE_LOOPDIR = 1

WAIT_SECONDS = 60


def tfrecord_dataset_with_source(files, source):
  return tf.data.TFRecordDataset(files).map(lambda value: (value, source))


def get_train_dataset(params):
  path = os.path.join(params.dataset_dir, 'train', 'train*')
  files = tf.gfile.Glob(path)
  if not files:
    raise ValueError('No training files found in %s' % path)
  return tfrecord_dataset_with_source(files, SOURCE_DATASETDIR)


def get_holparam_dataset(mode, params):
  """Create a Holparam dataset from train or test data.

  Optionally sample from fresh examples at a rate given by fresh_example_prob,
  and historical examples at a rate given by historical_example_prob.

  Args:
    mode: The mode for the input, one of the ModeKeys. Dataset is repeated in
      TRAIN mode.
    params: Hyperparameters for the input.

  Returns:
    dataset: A tf.data.Dataset object.
  """
  if mode == TRAIN:
    return get_train_dataset(params).repeat()

  if mode == EVAL:
    if params.eval_dataset_dir:
      path = os.path.join(params.eval_dataset_dir, 'valid*')
    else:
      path = os.path.join(params.dataset_dir, 'valid', 'valid*')
    files = tf.gfile.Glob(path)

    tf.logging.info('EVAL files: %s.', ' '.join([str(f) for f in files]))
    if not files:
      raise ValueError('No eval files found in %s' % path)

    return tfrecord_dataset_with_source(files, SOURCE_DATASETDIR)

  raise ValueError('Unrecognized mode %s' % mode)


def generic_parser(serialized_example, feature_list, label_list):
  """Parses a HOL example, keeping requested features and labels.

  Args:
    serialized_example: A tf.Example for a parameterized tactic application.
    feature_list: List of string feature names to parse (subset of features).
    label_list: List of string label names to parse (subset of labels).

  Returns:
    features, labels: dicts with keys of feature_list, label_list respectively.
  """
  example = tf.parse_single_example(
      serialized_example,
      features={
          # Subgoal features
          # goal: the consequent term of the subgoal as a string.
          'goal': tf.FixedLenFeature((), tf.string, default_value=''),
          # goal_asl: list of hypotheses of the subgoal.
          'goal_asl': tf.VarLenFeature(dtype=tf.string),
          # Parameterized tactic applied to the subgoal
          # tactic: string name of tactic that is applied to this subgoal.
          'tactic': tf.FixedLenFeature((), tf.string, default_value=''),
          # tac_id: integer id of tactic.
          'tac_id': tf.FixedLenFeature((), tf.int64, default_value=-1),
          # thms: list of tactic arguments of type thm.
          'thms': tf.VarLenFeature(dtype=tf.string),
          # thms_hard_negatives: list of hard negative theorem parameter
          # arguments
          'thms_hard_negatives': tf.VarLenFeature(dtype=tf.string),
      })

  for key in ('goal_asl', 'thms', 'thms_hard_negatives'):
    if key in example:
      example[key] = tf.sparse_tensor_to_dense(example[key], default_value='')

  features = {key: example[key] for key in feature_list}
  labels = {key: example[key] for key in label_list}
  return features, labels


def _choose_one_theorem_at_random(thms):
  """Adds tf ops to pick one theorem at random from a list of theorems."""
  size_of_thms = tf.size(thms)

  def get_an_element():
    random_index = tf.random_uniform([],
                                     minval=0,
                                     maxval=size_of_thms,
                                     dtype=tf.int32)
    return thms[random_index]

  return tf.cond(size_of_thms > 0, get_an_element, lambda: '')


def _shuffle_and_truncate_hard_negatives(thms_hard_negatives, params):
  """Adds tf ops to shuffle, truncate, and pad hard negatives with <NULL>."""
  shuffled_hard_negatives = tf.random.shuffle(thms_hard_negatives)
  slice_size = tf.math.minimum(
      tf.size(shuffled_hard_negatives), params.ratio_max_hard_negative_examples)
  truncated_hard_negatives = shuffled_hard_negatives[:slice_size]
  padding_size = params.ratio_neg_examples - slice_size
  padded_hard_negatives = tf.pad(
      truncated_hard_negatives, [[0, padding_size]], constant_values='<NULL>')
  return padded_hard_negatives


def pairwise_thm_parser(serialized_example, source, params):
  """Strips out a tactic id, goal term string, and random thm parameter.

  Args:
    serialized_example: A tf.Example for a parameterized tactic application.
    source: source of the example.
    params: Hyperparameters for the input.

  Returns:
    features['goal']: a string of the goal term.
    features['thms']: a string of a randomly chosen thm parameter or empty str.
    features['thms_hard_negatives']: list of strings, each a hard negative.
      Size controlled via params.
    labels['tac_id']: integer id of tactic applied.
  """
  del source  # unused

  feature_list = ['goal', 'thms', 'thms_hard_negatives']
  label_list = ['tac_id']
  features, labels = generic_parser(
      serialized_example, feature_list=feature_list, label_list=label_list)

  # thms: pick one uniformily at random
  features['thms'] = _choose_one_theorem_at_random(features['thms'])

  # thms_hard_negatives: Shuffle, truncate and then pad with '<NULL>'.
  features['thms_hard_negatives'] = _shuffle_and_truncate_hard_negatives(
      features['thms_hard_negatives'], params)

  return features, labels


def get_input_fn(dataset_fn,
                 mode,
                 params,
                 shuffle=None,
                 shuffle_queue=None,
                 repeat=None,
                 parser=None,
                 filt=None):
  """Create a HOL param input function getter.

  Args:
    dataset_fn: A function that generates the starting dataset object.
    mode: The mode for the input, one of the ModeKeys.
    params: Hyperparameters for the input.
    shuffle: Whether to shuffle the dataset.
    shuffle_queue: Size of the shuffle queue.
    repeat: Number of Epochs to repeat dataset. False does not repeat dataset.
      Default None value repeats input indefinitely only in TRAIN mode.
    parser: Function to use for parsing protos.
    filt: Filter function used to remove certain data.

  Returns:
    input_fn: A estimator input_fn.
  """

  if shuffle_queue is None:
    shuffle_queue = params.shuffle_queue
  if shuffle is None:
    shuffle = mode == TRAIN
  if repeat is None:
    do_repeat = mode == TRAIN
  elif not repeat:
    repeat = None
    do_repeat = False

  if parser is None:
    tf.logging.info('PASSED IN parser is None')
    parser = pairwise_thm_parser

  def input_fn():
    """Input Function for estimator."""
    ds = dataset_fn(params)
    if params.setdefault('cache', False):
      ds = ds.cache()
    if repeat is not None:
      ds = ds.repeat(repeat)
    elif do_repeat:
      ds = ds.repeat()
    if shuffle:
      ds = ds.shuffle(shuffle_queue)

    ds = ds.map(functools.partial(parser, params=params))

    if filt is not None:
      ds = ds.filter(filt)

    drop = mode == EVAL

    ds = ds.batch(params['batch_size'], drop_remainder=drop)
    return ds.make_one_shot_iterator().get_next()

  return input_fn
