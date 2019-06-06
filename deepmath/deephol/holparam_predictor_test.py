r"""Tests for holparam_predictor.

This test assumes an embedding size of 4.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags

import numpy as np
import tensorflow as tf
from deepmath.deephol import holparam_predictor
from deepmath.deephol import predictions_abstract_base_test
from deepmath.deephol import test_util

FLAGS = flags.FLAGS
NUM_TACTICS = 41
EMBEDDING_SIZE = 4

DEFAULT_BASE = 'deephol/test_data/default_ckpt/'
DEFAULT_TEST_PATH = os.path.join(DEFAULT_BASE, 'model.ckpt-0')
DEFAULT_SAVED_MODEL_PATH = os.path.join(
    DEFAULT_BASE, 'export/best_exporter/1548452515/saved_model.pb')

TAC_DEP_BASE = 'deephol/test_data/tac_dep_ckpt/'
TAC_DEP_TEST_PATH = os.path.join(TAC_DEP_BASE, 'model.ckpt-0')
TAC_DEP_SAVED_MODEL_PATH = os.path.join(
    TAC_DEP_BASE, 'export/best_exporter/1548452515/saved_model.pb')

DUMMY_BASE = '/path/to/checkpoints/'


class HolparamPredictorTest(
    predictions_abstract_base_test.PredictionsAbstractBaseTest):

  @classmethod
  def setUpClass(cls):
    """Restoring the graph takes a lot of time, so we do it only once here."""
    super(HolparamPredictorTest, cls).setUpClass()

    cls.checkpoint = test_util.test_src_dir_path(DEFAULT_TEST_PATH)
    cls.predictor = holparam_predictor.HolparamPredictor(cls.checkpoint)

  def _get_new_predictor(self):
    return holparam_predictor.HolparamPredictor(self.checkpoint)

  def _get_predictor(self):
    return self.predictor

  # TODO(smloos): move this function/test to predictions.py
  def testRecommendFromScores(self):
    tac_probs = [
        [0.1, 0.2, 0.3, 0.4],  # inverse argsort [3, 2, 1, 0]
        [0.4, 0.2, 0.1, 0.3]
    ]  # inverse argsort [0, 3, 1, 2]
    actual_rec = holparam_predictor.recommend_from_scores(tac_probs, 2)
    self.assertAllEqual([[3, 2], [0, 3]], actual_rec)

  def testGetSavedModelPath(self):
    test_pairs = [
        ((os.path.join(DUMMY_BASE,
                       'export/best_exporter/1557146333/variables/variables')),
         (os.path.join(DUMMY_BASE,
                       'export/best_exporter/1557146333/saved_model.pb'))),
        (test_util.test_src_dir_path(DEFAULT_TEST_PATH),
         test_util.test_src_dir_path(DEFAULT_SAVED_MODEL_PATH)),
        (test_util.test_src_dir_path(TAC_DEP_TEST_PATH),
         test_util.test_src_dir_path(TAC_DEP_SAVED_MODEL_PATH))
    ]
    for input_path, expected_path in test_pairs:
      actual_path = holparam_predictor.get_saved_model_path(input_path)
      self.assertEqual(expected_path, actual_path)


class TacticDependentPredictorTest(
    predictions_abstract_base_test.PredictionsAbstractBaseTest):

  @classmethod
  def setUpClass(cls):
    """Restoring the graph takes a lot of time, so we do it only once here."""
    super(TacticDependentPredictorTest, cls).setUpClass()

    cls.checkpoint = test_util.test_src_dir_path(TAC_DEP_TEST_PATH)
    cls.predictor = holparam_predictor.TacDependentPredictor(cls.checkpoint)

  def _get_new_predictor(self):
    return holparam_predictor.TacDependentPredictor(self.checkpoint)

  def _get_predictor(self):
    return self.predictor

  # Testing specific to Tactic Dependent Predictor
  def testTacticDependentBatchThmScores(self):
    predictor = self._get_predictor()
    emb1 = np.tile(1., EMBEDDING_SIZE)
    emb2 = np.tile(0.5, EMBEDDING_SIZE)
    [thm_score] = predictor.batch_thm_scores(emb1, [emb2], tactic_id=0)
    [thm_score_tac_id] = predictor.batch_thm_scores(emb1, [emb2], tactic_id=1)
    self.assertRaises(AssertionError, self.assertAlmostEqual, thm_score,
                      thm_score_tac_id)


if __name__ == '__main__':
  tf.test.main()
