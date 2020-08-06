r"""Tests for holparam_predictor.

This test assumes an embedding size of 4.
"""
import os
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
from deepmath.deephol import holparam_predictor
from deepmath.deephol import predictions
from deepmath.deephol import predictions_abstract_base_test
from deepmath.deephol import test_util

FLAGS = flags.FLAGS
NUM_TACTICS = 41
EMBEDDING_SIZE = 4

DEFAULT_BASE = 'deephol/test_data/default_ckpt/'
DEFAULT_TEST_PATH = os.path.join(DEFAULT_BASE, 'model.ckpt-0')

TAC_DEP_BASE = 'deephol/test_data/tac_dep_ckpt/'
TAC_DEP_TEST_PATH = os.path.join(TAC_DEP_BASE, 'model.ckpt-0')


class HolparamPredictorTest(
    predictions_abstract_base_test.PredictionsAbstractBaseTest):

  @classmethod
  def setUpClass(cls):
    """Restoring the graph takes a lot of time, so we do it only once here."""
    super(HolparamPredictorTest, cls).setUpClass()

    cls.checkpoint = test_util.test_src_dir_path(DEFAULT_TEST_PATH)
    cls.predictor = holparam_predictor.HolparamPredictor(cls.checkpoint)

  def _get_new_predictor(self, predictor_name):
    if predictor_name == 'holparam':
      return holparam_predictor.HolparamPredictor(self.checkpoint)
    raise ValueError('Unknown predictor name: %s' % predictor_name)

  def _get_predictor_map(self):
    return {'holparam': self.predictor}

  def testBatchPartnersLongInputs(self):
    # TODO(smloos): The wavenet model does not implement masking over padded
    # values, so we override this test. Once fixed, update the checkpoint and
    # remove this method.
    pass


class TacticDependentPredictorTest(
    predictions_abstract_base_test.PredictionsAbstractBaseTest):

  @classmethod
  def setUpClass(cls):
    """Restoring the graph takes a lot of time, so we do it only once here."""
    super(TacticDependentPredictorTest, cls).setUpClass()

    cls.checkpoint = test_util.test_src_dir_path(TAC_DEP_TEST_PATH)
    cls.predictor = holparam_predictor.TacDependentPredictor(cls.checkpoint)

  def _get_new_predictor(self, predictor_name):
    if predictor_name == 'tactic_dependent':
      return holparam_predictor.TacDependentPredictor(self.checkpoint)
    raise ValueError('Unknown predictor name: %s' % predictor_name)

  def _get_predictor_map(self):
    return {'tactic_dependent': self.predictor}

  # Testing specific to Tactic Dependent Predictor
  def testTacticDependentBatchThmScores(self):
    predictor = self._get_predictor_map()['tactic_dependent']
    emb1 = np.tile(1., EMBEDDING_SIZE)
    emb2 = np.tile(0.5, EMBEDDING_SIZE)
    [thm_score] = predictor.batch_thm_scores(emb1, [emb2], tactic_id=0)
    [thm_score_tac_id] = predictor.batch_thm_scores(emb1, [emb2], tactic_id=1)
    self.assertRaises(AssertionError, self.assertAlmostEqual, thm_score,
                      thm_score_tac_id)

  def testHolparamProofStateEmbedding(self):
    for predictor_name, predictor in self._get_predictor_map().items():
      for goal_proto in self.goal_protos:
        state = predictions.ProofState(goal=goal_proto)
        goal_emb = predictor.goal_proto_embedding(goal_proto)
        exp_state_emb = predictions.EmbProofState(goal_emb=goal_emb)
        actual_state_emb = predictor.proof_state_embedding(state)
        self.assertAllEqual(
            exp_state_emb.goal_emb,
            actual_state_emb.goal_emb,
            msg=('Proof state embeddings embeds the goal the same way as goal '
                 'proto embedding. Failed for %s on: %s' %
                 (predictor_name, goal_proto.conclusion)))

  def testBatchPartnersLongInputs(self):
    # TODO(smloos): The wavenet model does not implement masking over padded
    # values, so we override this test. Once fixed, update the checkpoint and
    # remove this method.
    pass


if __name__ == '__main__':
  tf.test.main()
