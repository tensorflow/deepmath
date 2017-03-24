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
"""Tests for clause_search.inputs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import numpy as np
import tensorflow as tf
from deepmath.eprover import prover_clause_examples_pb2
from deepmath.guidance import inputs

FLAGS = tf.flags.FLAGS


Info = collections.namedtuple('Info', ('conjecture', 'positives', 'negatives'))


class InputsTest(tf.test.TestCase):

  def testParseBuckets(self):
    self.assertAllEqual([2, 3, 4], inputs.parse_buckets('2,3,4'))

  def testExamplesShards(self):
    shards = 6
    FLAGS.examples_train = '/blah/examples-train@%d' % shards
    FLAGS.examples_eval = '/blah/examples-eval@%d' % shards
    for mode in 'train', 'eval':
      self.assertAllEqual(inputs.examples_shards(mode=mode),
                          ['/blah/examples-%s-%05d-of-%05d' % (mode, i, shards)
                           for i in range(shards)])

  def testReadVocab(self):
    # Mirrors tests in vocabulary_test.cc
    path = os.path.join(self.get_temp_dir(), 'small_vocab')
    with open(path, 'w') as vocab_file:
      for s in '7', 'X', 'Yx', 'f', 'g':
        print(s, file=vocab_file)

    def check(vocab_to_id, expect):
      expect.update({' ': 0, '*': 1, '~': 2, '|': 3, '&': 4, '(': 5, ')': 6,
                     ',': 7, '=': 8, '$false': 9, '$true': 10})
      for word, i in expect.items():
        self.assertEqual(vocab_to_id[word], i)

    # No flags
    size, vocab_to_id = inputs.read_vocab(path)
    self.assertEqual(size, 32 + 5)
    check(vocab_to_id,
          {'7': 32 + 0, 'X': 32 + 1, 'Yx': 32 + 2, 'f': 32 + 3, 'g': 32 + 4})

    # One variable
    size, vocab_to_id = inputs.read_vocab(path + ':one_variable')
    self.assertEqual(size, 32 + 4)
    check(vocab_to_id,
          {'7': 32 + 0, 'X': 32 + 1, 'Yx': 32 + 1, 'f': 32 + 2, 'g': 32 + 3})

  def testProtoBatch(self):
    shards = 10
    batch_size = 4
    examples_per_shard = 6
    FLAGS.examples_train = os.path.join(self.get_temp_dir(),
                                        'examples-train@%d' % shards)
    FLAGS.examples_eval = os.path.join(self.get_temp_dir(),
                                       'examples-eval@%d' % shards)
    FLAGS.approx_proofs_per_shard = examples_per_shard
    FLAGS.input_queue_factor = 2

    # Write sharded tfrecords
    mode_values = {'train': set(), 'eval': set()}
    for mode in 'train', 'eval':
      for shard in range(shards):
        shard_path = os.path.join(
            self.get_temp_dir(),
            'examples-%s-%05d-of-%05d' % (mode, shard, shards))
        with tf.python_io.TFRecordWriter(shard_path) as writer:
          for i in range(examples_per_shard):
            value = tf.compat.as_bytes('value-%s-%d.%d' % (mode, shard, i))
            writer.write(value)
            mode_values[mode].add(value)

    def traverse(mode, epochs, shuffle):
      """Record the keys seen throughout some number of epochs."""
      with tf.Graph().as_default() as graph:
        tf.set_random_seed(7)
        values = inputs.proto_batch(mode=mode, batch_size=batch_size,
                                    shuffle=shuffle)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        with self.test_session(graph=graph):
          init_op.run()
          coord = tf.train.Coordinator()
          threads = tf.train.start_queue_runners(coord=coord)
          counts = collections.defaultdict(int)
          for _ in range(epochs * shards * examples_per_shard // batch_size):
            values_np = values.eval()
            self.assertEqual(values_np.shape, (batch_size,))
            for value in values_np:
              self.assertIn(value, mode_values[mode])
              counts[value] += 1
          coord.request_stop()
          for thread in threads:
            thread.join()
          return counts

    for mode in 'train', 'eval':
      # With shuffling off, we should be able to hit each example once
      counts = traverse(mode, epochs=1, shuffle=False)
      for value in mode_values[mode]:
        self.assertEqual(counts[value], 1)

      # With shuffling on, the counts will be off but should have the right sum
      epochs = 10
      counts = traverse(mode, epochs=epochs, shuffle=True)
      self.assertEqual(np.sum(list(counts.values())),
                       epochs * len(mode_values[mode]))

  def testSequence(self):
    # Random generation of ProverClauseExamples
    vocab = set()

    def random_list(limit, empty, separator, f):
      count = np.random.randint(limit)
      if not count:
        return empty
      return separator.join(f() for _ in range(count))

    def new_name(prefix):
      s = '%s%d' % (prefix, len(vocab))
      vocab.add(s)
      return s

    def random_term(term, depth):
      if depth == 0 or np.random.randint(3) == 0:
        if np.random.randint(2):
          name = term.variable.name = new_name('X')
          return name
        else:
          name = term.number.value = new_name('')
          return name
      else:
        name = term.function.name = new_name('f')

        def random_arg():
          return random_term(term.function.args.add(), depth=depth - 1)

        args = random_list(2, '', ',', random_arg)
        return '%s(%s)' % (name, args) if args else name

    def random_equation(equation):
      equation.negated = np.random.randint(2)
      s = '~' * equation.negated
      s += random_term(equation.left, depth=2)
      if np.random.randint(2):
        s += '=' + random_term(equation.right, depth=1)
      return s

    def random_clause(clause):
      return random_list(4, '$false', '|',
                         lambda: random_equation(clause.clause.equations.add()))

    def random_clauses(clauses):
      return random_list(4, '$true', '&',
                         lambda: '(%s)' % random_clause(clauses.add()))

    np.random.seed(7)
    tf.set_random_seed(7)
    shards = 10
    batch_size = 2
    examples_per_shard = 6
    FLAGS.examples_train = os.path.join(self.get_temp_dir(),
                                        'examples-train@%d' % shards)
    FLAGS.examples_eval = os.path.join(self.get_temp_dir(),
                                       'examples-eval@%d' % shards)
    FLAGS.approx_proofs_per_shard = examples_per_shard
    FLAGS.input_queue_factor = 2

    # Build tfrecords of ProverClauseExamples
    key_info = {}
    mode_keys = {'train': set(), 'eval': set()}
    valid_keys = set()  # Keys with at least one positive and negative
    for mode in 'train', 'eval':
      for shard in range(shards):
        shard_path = os.path.join(
            self.get_temp_dir(),
            'examples-%s-%05d-of-%05d' % (mode, shard, shards))
        with tf.python_io.TFRecordWriter(shard_path) as writer:
          valid_count = 0
          while valid_count < examples_per_shard:
            key = 'key%d' % len(key_info)
            full_key = tf.compat.as_bytes('%s:%s' % (shard_path, key))
            examples = prover_clause_examples_pb2.ProverClauseExamples()
            examples.key = full_key
            conjecture = random_clauses(examples.cnf.negated_conjecture)
            positives = [random_clause(examples.positives.add())
                         for _ in range(np.random.randint(3))]
            negatives = [random_clause(examples.negatives.add())
                         for _ in range(np.random.randint(3))]
            writer.write(examples.SerializeToString())
            key_info[full_key] = Info(conjecture, positives, negatives)
            if positives and negatives:
              mode_keys[mode].add(full_key)
              valid_keys.add(full_key)
              valid_count += 1

    # Write vocab file
    vocab_path = os.path.join(self.get_temp_dir(), 'vocab')
    with open(vocab_path, 'w') as vocab_file:
      for s in vocab:
        print(s, file=vocab_file)
    FLAGS.vocab = vocab_path

    # Read vocabulary, and construct map from int sequence back to string
    vocab_size, vocab_to_id = inputs.read_vocab(vocab_path)
    self.assertEqual(vocab_size, len(vocab_to_id) + 32 - 11)
    id_to_vocab = {i: s for s, i in vocab_to_id.items()}

    def show_ids(ids):
      """Converts a coded clause to string, truncating and stripping."""
      return ''.join(id_to_vocab[i] for i in ids).strip()

    # Test both train and eval
    for shuffle in False, True:
      if shuffle:
        buckets = '16,32,64,128,256,512'
      else:
        # Disable bucketing so that we can verify everything is processed
        buckets = '100000'
      FLAGS.negated_conjecture_buckets = FLAGS.clause_buckets = buckets
      for mode in 'train', 'eval':
        with tf.Graph().as_default() as graph:
          keys, conjectures, clauses, labels = (inputs.sequence_example_batch(
              mode=mode, batch_size=batch_size, shuffle=shuffle))
          init_op = tf.group(tf.global_variables_initializer(),
                             tf.local_variables_initializer())
          self.assertEqual(keys.dtype, tf.string)
          self.assertEqual(conjectures.dtype, tf.int32)
          self.assertEqual(clauses.dtype, tf.int32)
          self.assertEqual(labels.dtype, tf.bool)

          # Evaluate enough times to see every key exactly twice
          with self.test_session(graph=graph) as sess:
            init_op.run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            visited = collections.defaultdict(int)
            for _ in range(len(mode_keys[mode]) // batch_size):
              batch = sess.run([keys, conjectures, clauses, labels])
              for data in batch:
                self.assertEqual(len(data), batch_size)
              for pair in batch[2:]:
                self.assertEqual(pair.shape[1], 2)
              for key, conjecture, clause_pair, label_pair in zip(*batch):
                self.assertIn(key, mode_keys[mode],
                              'mode %s, key %r, keys %r' %
                              (mode, key, mode_keys[mode]))
                visited[key] += 1
                info = key_info[key]
                self.assertEqual(info.conjecture, show_ids(conjecture))
                for clause, label in zip(clause_pair, label_pair):
                  self.assertIn(show_ids(clause),
                                info.positives if label else info.negatives)
            coord.request_stop()
            for thread in threads:
              thread.join()

        if not shuffle:
          # Verify that we visited everything exactly twice
          for key in mode_keys[mode]:
            count = visited[key]
            if count != 1:
              raise ValueError('key %s visited %d != 1 times' % (key, count))

  def testDepth(self):
    # Build very simple vocabulary
    FLAGS.vocab = os.path.join(self.get_temp_dir(), 'depth_vocab')
    with open(FLAGS.vocab, 'w') as vocab_file:
      print('X\nf', file=vocab_file)
    _, vocab_to_id = inputs.read_vocab(FLAGS.vocab)

    # Build two very deep clauses
    def deep_clause(n, clause):
      term = clause.clause.equations.add().left
      for _ in range(n):
        term.function.name = 'f'
        term = term.function.args.add()
      term.variable.name = 'X'
    examples = prover_clause_examples_pb2.ProverClauseExamples()
    deep_clause(100, examples.positives.add())
    deep_clause(200, examples.negatives.add())

    # The clause are f(f(...(X)...))
    def correct(n):
      correct = ['f', '('] * n + ['X'] + [')'] * n
      return [vocab_to_id[s] for s in correct]

    # Check that parsing works
    with self.test_session() as sess:
      _, negated_conjecture, clauses, labels = sess.run(
          inputs.random_clauses_as_sequence(examples.SerializeToString(),
                                            vocab=FLAGS.vocab))
      def decode(s):
        return np.fromstring(s, dtype=np.int32)
      self.assertAllEqual(decode(negated_conjecture), [vocab_to_id['$true']])
      self.assertAllEqual(decode(clauses[0]), correct(100))
      self.assertAllEqual(decode(clauses[1]), correct(200))
      self.assertAllEqual(labels, [True, False])


if __name__ == '__main__':
  tf.test.main()
