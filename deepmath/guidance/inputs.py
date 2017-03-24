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
"""Input pipeline for clause search models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf
from deepmath.guidance import gen_clause_ops
from deepmath.guidance import jagged
from deepmath.util import dfile

slim = tf.contrib.slim
flags = tf.flags
FLAGS = flags.FLAGS

# Data sources
flags.DEFINE_string(
    'examples_train',
    '/readahead/32M/cns/oo-d/home/geoffreyi/mizar/moreproofs/'
    '00adep15_proofs_examples_train@160',
    'Training examples as a ProverClauseExamples sstable.')
flags.DEFINE_string(
    'examples_eval',
    '/readahead/32M/cns/oo-d/home/geoffreyi/mizar/moreproofs/'
    '00adep15_proofs_examples_eval@160',
    'Evaluation examples as a ProverClauseExamples sstable.')
flags.DEFINE_string('vocab',
                    '/cns/oo-d/home/geoffreyi/mizar/00adep15_vocab.txt',
                    'Path to vocabulary file.')

# Batching, bucketing, queues
flags.DEFINE_integer('approx_proofs_per_shard', 205,
                     'Approximate (ideally upper bound) proofs per shard.')
flags.DEFINE_integer('input_queue_factor', 32,
                     'Higher numbers shuffle across more proofs.')
flags.DEFINE_string('negated_conjecture_buckets', '64,128,256,512,1024',
                    'Bucket sizes for negated conjectures.')
flags.DEFINE_string('clause_buckets', '32,64,128,256,512',
                    'Bucket sizes for clauses (either positive or negative).')


def parse_buckets(spec):
  """Parse a comma separated list of bucket sizes.

  Args:
    spec: Comma separated list of bucket sizes.

  Returns:
    Tuple of bucket size sizes.
  """
  return np.array([int(s) for s in spec.split(',')])


def examples_shards(mode):
  """Pick which examples shards to use.

  Args:
    mode: Either 'train' or 'eval'.

  Returns:
    A list of filenames.

  Raises:
    ValueError: If mode is not 'train' or 'eval'.
  """
  if mode == 'train':
    path = FLAGS.examples_train
  elif mode == 'eval':
    path = FLAGS.examples_eval
  else:
    raise ValueError('Invalid mode %r' % mode)
  return dfile.sharded_filenames(path)


def read_vocab(path):
  """Read vocabulary from an sstable.

  Must be kept synchronized with the Vocabulary class in clause_ops.cc.
  Vocabulary flags may be specified after a ':'; for now the only valid
  flag is ':one_variable'.

  Args:
    path: Path to vocabulary sstable.

  Returns:
    size: Vocabulary size (maximum id + 1).
    vocab_to_id: Dictionary mapping symbols and vocab words to integers.

  Raises:
    ValueError: If path contains an invalid flag.
  """
  # Remove flags from path
  if ':' in path.split('/')[-1]:
    path = path.split(':')
    options = path[-1].split(',')
    path = ':'.join(path[:-1])
    valid_options = ('one_variable',)
    for f in options:
      if f not in valid_options:
        raise ValueError("Invalid vocabulary flag '%s'" % f)
  else:
    options = ()
  one_variable = 'one_variable' in options

  # Generate vocabulary
  symbols = tuple(' *~|&(),=') + ('$false', '$true')
  vocab_to_id = {}
  for i, s in enumerate(symbols):
    vocab_to_id[s] = i
  next_id = 32
  variable_id = None
  for s in tf.gfile.Open(path):
    s = s.strip()
    if not s: continue
    if one_variable and s[:1].isupper():
      if variable_id is None:
        variable_id = next_id
        next_id += 1
      i = variable_id
    else:
      i = next_id
      next_id += 1
    vocab_to_id[s] = i
  size = 1 + max(vocab_to_id.values())
  return size, vocab_to_id


def random_clauses_as_sequence(examples, vocab, seed=None, name=None):
  """Select one positive and one negative clause from a ProverClauseExamples.

  If the ProverClauseExamples doesn't have both positives and negatives,
  clauses and labels will be empty.

  Args:
    examples: 0-D `string` `Tensor`.  Serialized ProverClauseExamples.
    vocab: Path to vocabulary sstable.
    seed: A Python integer. Used to create a random seed for the distribution.
    name: A name for the operation (optional).

  Returns:
    negated_conjecture: 0-D `int32` negated encoded conjecture as `string`.
    clauses: 1-D `int32` clauses encoded as `string`.  One positive, one
      negative if possible, otherwise empty.
    labels: 1-D `bool` labels (true for positive, false for negative).
  """
  seed1, seed2 = tf.get_seed(seed)
  return gen_clause_ops.random_clauses_as_sequence(examples=examples,
                                                   vocab=vocab,
                                                   seed=seed1,
                                                   seed2=seed2,
                                                   name=name)


def fast_clauses_as_sequence_jagged(clauses, name=None):
  """Serialize FastClause protos into a jagged tensor of ids.

  Args:
    clauses: 1-D tensor or 2-D jagged tensor of FastClause protos.
    name: Optional name for this operation.

  Returns:
    Jagged tensor of id sequences.
  """
  sizes, flat = jagged.unjagged(clauses)
  sizes_flat, flat = gen_clause_ops.fast_clauses_as_sequence_jagged(
      flat, name=name)
  return jagged.jagged(jagged.jagged(sizes, sizes_flat), flat)


def proto_batch(mode, batch_size, shuffle=True):
  """Make a queue of serialized ProverClauseExamples protos.

  Args:
    mode: Either 'train' or 'eval'.
    batch_size: Integer batch size.
    shuffle: Whether to shuffle.

  Returns:
    A queue of serialized ProverClauseExamples.
  """
  # Generate ProverClauseExamples
  _, examples = dfile.read_sstable_or_tfrecord(examples_shards(mode),
                                               shuffle=shuffle)

  # Make a shuffling or fifo queue as requested
  capacity = 4 * batch_size
  dtypes = tf.string,
  shapes = ()
  if shuffle:
    # We only need to shuffle enough to mix across many different
    # ProverClauseExamples, since the sstable is assumed preshuffled.  The
    # factor of 2 accounts for positives and negatives.
    extra = 2 * FLAGS.approx_proofs_per_shard * FLAGS.input_queue_factor
    example_queue = tf.RandomShuffleQueue(capacity=extra + capacity,
                                          min_after_dequeue=extra,
                                          dtypes=dtypes,
                                          shapes=shapes,
                                          name='examples_queue')
  else:
    example_queue = tf.FIFOQueue(capacity=capacity,
                                 dtypes=dtypes,
                                 shapes=shapes,
                                 name='examples_queue')
  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
      example_queue, [example_queue.enqueue([examples])]))

  # Grab one batch at a time
  return example_queue.dequeue_many(batch_size)


def batch_pad(sequences):
  """Pad with zeros out to maximum length of the batch of sequences.

  Args:
    sequences: 1-D integer sequences to pad.

  Returns:
    2-D padded int32 numpy array.
  """
  length = max(len(s) for s in sequences)
  batch = np.zeros((len(sequences), length), dtype=np.int32)
  for i, s in enumerate(sequences):
    batch[i, :len(s)] = s
  return batch


def sequence_example_batch(mode, batch_size, shuffle=True):
  """Generates a padded batch of (negated_conjecture, clause, label) examples.

  Each batch is organized into (positive, negative) pairs of examples from the
  same proof.

  Args:
    mode: Either 'train' or 'eval'.
    batch_size: Integer batch size.
    shuffle: Whether to shuffle.

  Returns:
    keys: 1-D `(batch_size,)` `string` tensor.
    negated_conjectures: 2-D `(batch_size, ?)` `int32` tensor.
    clauses: 2-D `(batch_size, 2, ?)` `int32` tensor.
    labels: 1-D `(batch_size, 2)` `bool` tensor.
  """
  # Generate ProverClauseExamples.
  _, examples = dfile.read_sstable_or_tfrecord(examples_shards(mode),
                                               shuffle=shuffle)

  # Parse buckets
  negated_conjecture_buckets = parse_buckets(FLAGS.negated_conjecture_buckets)
  clause_buckets = parse_buckets(FLAGS.clause_buckets)

  # Bucket examples by length and pad
  unbucketed = collections.deque()
  buckets = {(len_a, len_b): []
             for len_a in negated_conjecture_buckets
             for len_b in clause_buckets}

  # Determine number of examples needed in a single dequeue.
  def examples_needed():
    """Computes the minimum number of examples for bucket_and_pad to work."""
    need = len(buckets) * (batch_size - 1) + 1 - len(unbucketed)
    return np.int32(max(0, need))

  examples_needed = tf.py_func(examples_needed, [], [tf.int32],
                               name='examples_needed')

  # Make a shuffling or fifo queue as requested.
  capacity = (4 * batch_size * len(negated_conjecture_buckets) *
              len(clause_buckets))
  shapes = (), (), (2,), (2,)
  # Extract one positive and one negative from each ProverClauseExamples.
  key, conjecture, clauses, labels = random_clauses_as_sequence(
      examples, vocab=FLAGS.vocab)
  if shuffle:
    # We only need to shuffle enough to mix across many different
    # ProverClauseExamples, since the sstable is assumed preshuffled.
    extra = FLAGS.approx_proofs_per_shard * FLAGS.input_queue_factor
    keys, conjectures, clauses, labels = tf.train.maybe_shuffle_batch(
        [key, conjecture, clauses, labels],
        batch_size=examples_needed,
        capacity=extra + capacity,
        min_after_dequeue=extra,
        keep_input=tf.size(clauses) > 0,
        shapes=shapes,
        name='examples_queue')
  else:
    keys, conjectures, clauses, labels = tf.train.maybe_batch(
        [key, conjecture, clauses, labels],
        keep_input=tf.size(clauses) > 0,
        batch_size=examples_needed,
        capacity=capacity,
        shapes=shapes,
        name='examples_queue')

  def find_bucket(data, buckets):
    """Find bucket for data and return (truncated, bucket)."""
    data = np.fromstring(data, dtype=np.int32)
    data = data[:buckets[-1]]  # Truncate if necessary
    bucket = np.min(buckets[len(data) <= buckets])
    return data, bucket

  def bucket_and_pad(keys, conjectures, clauses, labels):
    unbucketed.extend(zip(keys, conjectures, clauses, labels))
    # Loop until one bucket is full, then return it as a batch
    while True:
      key, conjecture, clauses, labels = unbucketed.popleft()
      conjecture, b0 = find_bucket(conjecture, negated_conjecture_buckets)
      clauses, b1s = zip(*[find_bucket(c, clause_buckets) for c in clauses])
      bucket = buckets[(b0, max(b1s))]
      bucket.append((key, conjecture, clauses, labels))
      if len(bucket) == batch_size:
        keys, conjectures, clauses, labels = zip(*bucket)
        del bucket[:]
        conjectures = batch_pad(conjectures)
        clauses = batch_pad([c for cs in clauses for c in cs])
        return keys, conjectures, clauses.reshape(batch_size, 2, -1), labels

  keys, conjectures, clauses, labels = tf.py_func(
      bucket_and_pad, [keys, conjectures, clauses, labels],
      [tf.string, tf.int32, tf.int32, tf.bool],
      name='bucket_and_pad')

  # Set shapes to make Keras happy
  keys.set_shape([batch_size])
  conjectures.set_shape([batch_size, None])
  clauses.set_shape([batch_size, 2, None])
  labels.set_shape([batch_size, 2])

  return keys, conjectures, clauses, labels
