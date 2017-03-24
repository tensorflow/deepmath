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
"""Tests for clause_search.driver_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import keras
import numpy as np
import tensorflow as tf
from deepmath.eprover import fast_clause_pb2
from deepmath.guidance import driver_lib
from deepmath.util import test_utils

FLAGS = tf.flags.FLAGS


def random_fast_clause():

  def random_term(term, depth=2):
    term.id = np.random.randint(100)
    if depth:
      for _ in range(np.random.randint(3)):
        random_term(term.args.add(), depth=depth - 1)

  clause = fast_clause_pb2.FastClause()
  for _ in range(np.random.randint(3)):
    equation = clause.equations.add()
    equation.negated = np.random.randint(2)
    random_term(equation.left)
    if np.random.randint(2):
      random_term(equation.right)
  return clause.SerializeToString()


class DriverTest(tf.test.TestCase):

  def testParseHParams(self):
    hparams = driver_lib.parse_hparams('model=model_definition_cnn_flat3')
    self.assertEqual(hparams.model, 'model_definition_cnn_flat3')
    for s in ('model=model_cnn_regularized,keep_prob=0.5',
              'keep_prob=0.5,model=model_cnn_regularized'):
      hparams = driver_lib.parse_hparams(s)
      self.assertEqual(hparams.model, 'model_cnn_regularized')
      self.assertEqual(hparams.keep_prob, 0.5)
      self.assertEqual(hparams.batch_norm, False)

  def testModeBatchSize(self):
    hparams = driver_lib.parse_hparams(
        'model=model_cnn_regularized,batch_size=7,eval_batch_size=11')
    self.assertEqual(driver_lib.mode_batch_size('train', hparams), 7)
    self.assertEqual(driver_lib.mode_batch_size('eval', hparams), 11)


class ModelsTest(tf.test.TestCase):

  def _modelTest(self, hparam_str):
    with test_utils.temp_dir() as tmp:
      # Configure
      test_data = 'deepmath/guidance/test_data'
      FLAGS.examples_train = os.path.join(test_data, 'examples-train@10')
      FLAGS.examples_eval = os.path.join(test_data, 'examples-eval@10')
      FLAGS.vocab = os.path.join(test_data, 'vocab')
      FLAGS.approx_proofs_per_shard = 4  # Lie since lower means fast tests
      FLAGS.input_queue_factor = 1
      FLAGS.logdir = tmp
      hparams = driver_lib.parse_hparams(hparam_str)
      hparams.seed = 13

      # Train
      keras.backend.clear_session()
      hparams.batch_size = 2
      hparams.max_steps = 1
      with tf.Graph().as_default():
        driver_lib.run_mode('train', hparams)

      # Eval
      keras.backend.clear_session()
      hparams.eval_batch_size = 3
      hparams.eval_examples = 3
      hparams.max_evals = 1
      with tf.Graph().as_default():
        driver_lib.run_mode('eval', hparams)

      # Infer
      keras.backend.clear_session()
      meta = os.path.join(tmp, 'graph.meta')
      checkpoint = tf.train.latest_checkpoint(os.path.join(tmp, 'train'))
      driver_lib.export_inference_meta_graph(hparams, filename=meta)
      with tf.Graph().as_default() as graph:
        np.random.seed(5)
        with self.test_session(graph=graph) as sess:
          saver = tf.train.import_meta_graph(meta)
          saver.restore(sess, checkpoint)
          # Check names
          conjecture = graph.get_tensor_by_name('conjecture:0')
          clauses = graph.get_tensor_by_name('clauses:0')
          conjecture_embedding = graph.get_tensor_by_name(
              'conjecture_embeddings:0')
          logits = graph.get_tensor_by_name('logits:0')
          graph.get_operation_by_name('initialize')
          # Verify that we can precompute conjecture embeddings and then
          # classify.
          batch_size = 3
          np_conjecture = [random_fast_clause() for _ in range(2)]
          np_clauses = [random_fast_clause() for _ in range(batch_size)]
          conjecture = conjecture_embedding.eval(
              feed_dict={conjecture: np_conjecture})
          self.assertEqual(conjecture.shape[:-1], (1,))
          logits = logits.eval(feed_dict={conjecture_embedding: conjecture,
                                          clauses: np_clauses})
          self.assertEqual(logits.shape, (batch_size,))

  def testPremiseCNN(self):
    self._modelTest('model=model_definition_cnn_flat3')

  def testPremiseCNNLSTM(self):
    self._modelTest('model=model_final_cnn_3x_lstm')

  def testCNNRegularizedDropout(self):
    self._modelTest('model=model_cnn_regularized,keep_prob=0.5,filter_width=3')

  def testCNNRegularizedBatchNorm(self):
    self._modelTest('model=model_cnn_regularized,batch_norm=true,'
                    'filter_width=3')

  def testCNNRegularizedDecay(self):
    self._modelTest('model=model_cnn_regularized,weight_decay=0.0001,'
                    'filter_width=3')

  def testWavenet(self):
    self._modelTest('model=model_wavenet')

  def testTreeRNN(self):
    self._modelTest('model=tree_rnn,cell=rnn-relu,embedding_size=29,'
                    'hidden_size=57')

  def testDeepTreeRNN(self):
    self._modelTest('model=tree_rnn,cell=rnn-relu,layers=3,embedding_size=15,'
                    'hidden_size=33')

  def testTreeLSTM(self):
    self._modelTest('model=tree_rnn,cell=lstm,embedding_size=29,'
                    'hidden_size=57,keep_prob=0.5')

  def testDeepTreeLSTM(self):
    self._modelTest('model=tree_rnn,cell=lstm,layers=3,embedding_size=7,'
                    'hidden_size=9')

  def testTreeJoint(self):
    self._modelTest('model=tree_rnn,cell=rnn-relu,embedding_size=16,'
                    'hidden_size=16,loss=joint')

  def testPremiseCNNJoint(self):
    self._modelTest('model=model_definition_cnn_flat3,loss=joint')

  def testCNNUnconditional(self):
    self._modelTest('model=cnn_unconditional,hidden_size=7,inner_size=5,'
                    'classifier_size=3')

  def testFastCNN(self):
    self._modelTest('model=fast_cnn,hidden_size=7')


if __name__ == '__main__':
  tf.test.main()
