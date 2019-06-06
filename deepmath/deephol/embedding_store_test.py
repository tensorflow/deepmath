"""Tests for third_party.deepmath.deephol.embedding_store_lib."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from absl import flags
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from deepmath.deephol import embedding_store
from deepmath.deephol import io_util
from deepmath.deephol import mock_predictions_lib
from deepmath.deephol import test_util
from deepmath.deephol.utilities import normalization_lib
from deepmath.proof_assistant import proof_assistant_pb2

TEST_THEOREM_DB_PATH = 'deephol/data/mini_theorem_database.textpb'
MOCK_PREDICTOR = mock_predictions_lib.MockPredictionsLib()


def _process_thms(thms):
  return [normalization_lib.normalize(thm).conclusion for thm in thms]


class EmbeddingStoreTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    self.test_subdirectory = tempfile.mkdtemp(dir=flags.FLAGS.test_tmpdir)
    self.store = embedding_store.TheoremEmbeddingStore(MOCK_PREDICTOR)
    self.thm_db = proof_assistant_pb2.TheoremDatabase()
    for i in range(8):
      thm = self.thm_db.theorems.add()
      thm.conclusion = 'th%d' % i
    self.thms = _process_thms(self.thm_db.theorems)
    self.assertLen(self.thms, 8)

  def test_init_embedding_store(self):
    self.assertEqual(self.store.predictor, MOCK_PREDICTOR)
    self.assertIsNone(self.store.thm_embeddings)

  def test_compute_embeddings_for_thms_from_db(self):
    store = self.store
    store.compute_embeddings_for_thms_from_db(self.thm_db)
    self.assertAllClose(store.thm_embeddings,
                        np.array([[thm.__hash__(), 1] for thm in self.thms]))

  def test_compute_thm_embeddings_for_thms_from_file(self):
    store = self.store
    self.assertIsNone(store.thm_embeddings)
    file_path = test_util.test_src_dir_path(TEST_THEOREM_DB_PATH)
    store.compute_embeddings_for_thms_from_db_file(file_path)
    db = io_util.load_theorem_database_from_file(file_path)
    self.thms = _process_thms(db.theorems)
    self.assertAllClose(store.thm_embeddings,
                        np.array([[thm.__hash__(), 1] for thm in self.thms]))

  def test_save_read_embeddings(self):
    store = self.store
    store.compute_embeddings_for_thms_from_db(self.thm_db)
    file_path = os.path.join(flags.FLAGS.test_tmpdir, 'embs', 'embs.npy')
    store.save_embeddings(file_path)
    store2 = embedding_store.TheoremEmbeddingStore(MOCK_PREDICTOR)
    store2.read_embeddings(file_path)
    self.assertAllClose(store.thm_embeddings, store2.thm_embeddings)

  @parameterized.parameters(1, 3, 7, None)
  def test_get_thm_scores_for_preceding_thms(self, theorem_index):
    store = self.store
    store.compute_embeddings_for_thms_from_db(self.thm_db)
    goal_embedding = np.array([1.0, 2.0])
    if theorem_index is not None:
      test_theorem_index = theorem_index
    else:
      test_theorem_index = len(self.thms)
    scores = store.get_thm_scores_for_preceding_thms(goal_embedding,
                                                     theorem_index)
    expected_thm_scores = np.array([
        3.0 + 2.0 * (1.0 + thm.__hash__())
        for thm in self.thms[:test_theorem_index]
    ])
    self.assertAllClose(scores[:len(expected_thm_scores)], expected_thm_scores)

  def test_compute_assumption_embeddings(self):
    assumptions = ['(v A x)', '(v A y)']
    self.assertRaises(NotImplementedError,
                      self.store.compute_assumption_embeddings, assumptions)


if __name__ == '__main__':
  tf.test.main()
