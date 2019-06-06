"""Abstract test suite for instantiations of third_party.deepmath.deephol.predictions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from absl import flags
import numpy as np
import six
import tensorflow as tf
from deepmath.deephol import predictions

FLAGS = flags.FLAGS
NUM_TACTICS = 41
EMBEDDING_SIZE = 4


class PredictionsAbstractBaseTest(
    six.with_metaclass(abc.ABCMeta, tf.test.TestCase)):

  @classmethod
  @abc.abstractmethod
  def setUpClass(cls):
    """Concrete instances must create cls.predictor."""
    pass

  def setUp(self):
    super(PredictionsAbstractBaseTest, self).setUp()

    self.formulas = ['(v A x)', '(v A y)', '(= a b)', '']
    # TODO(adipal): Add back empty string test when GNNs support it.
    self.goal_formulas = self.formulas[:-1]
    self.embeddings = [
        np.arange(float(i), float(i + EMBEDDING_SIZE))
        for i in range(len(self.formulas))
    ]
    self.tactic_id = 0

  @abc.abstractmethod
  def _get_new_predictor(self):
    pass

  @abc.abstractmethod
  def _get_predictor(self):
    pass

  def _testSameEmbeddings(self, formulas):
    """Test that the same formulas are embedded to the same vector."""
    predictor = self._get_predictor()
    goal_embs = predictor.batch_goal_embedding(formulas)
    # Goal embeddings must be the same for the same goal string.
    self.assertAllCloseAccordingToType(goal_embs[0], goal_embs[1])

    # Theorem embeddings must be the same for the same theorem string.
    thm_embs = predictor.batch_thm_embedding(formulas)
    self.assertAllCloseAccordingToType(thm_embs[0], thm_embs[1])

    # Theorem and goal embeddings should not be the same.
    for ge, te in zip(goal_embs, thm_embs):
      self.assertRaises(AssertionError, self.assertAllCloseAccordingToType, ge,
                        te)

    # Tactic predictor should all be the same.
    tac_pred = predictor.batch_tactic_scores(goal_embs)
    self.assertAllCloseAccordingToType(
        tac_pred[0],
        tac_pred[1],
        msg='Tactic scores must be the same for the same goal embedding.')

    # Tactic predictor should give a probability for each tactic.
    self.assertLen(
        tac_pred[0], NUM_TACTICS,
        'A probability should be predicted for each tactic (%d).' % NUM_TACTICS)
    self.assertLen(
        tac_pred[1], NUM_TACTICS,
        'A probability should be predicted for each tactic (%d).' % NUM_TACTICS)

    self.assertAllCloseAccordingToType(
        tac_pred,
        predictor.batch_tactic_scores(goal_embs),
        msg='Tactic scores should be the same, even with multiple executions.')

  def testSameEmbeddingsExact(self):
    self._testSameEmbeddings(['(v A x)', '(v A x)'])

  def testSameEmbeddingsObscure(self):
    self._testSameEmbeddings([
        '(a not_in_vocab_1 not_in_vocab2)', '(a not_in_vocab_3 not_in_vocab4)'
    ])

  def testDifferentEmbedding(self):
    """Goals are different. Permuting batch or reloading graph has no effect."""
    predictor = self._get_predictor()
    goal_embs = predictor.batch_goal_embedding(self.goal_formulas)
    thm_embs = predictor.batch_thm_embedding(self.formulas)

    self.assertLen(
        set([tuple(emb) for emb in goal_embs]), len(self.goal_formulas),
        'Unique goals should be uniquely embedded.')
    self.assertLen(
        set([tuple(emb) for emb in thm_embs]), len(self.formulas),
        'Unique theorems should be uniquely embedded.')

  def _assert_all_close_batch(self, batch1, batch2, message):
    for v1, v2 in zip(batch1, batch2):
      self.assertAllClose(v1, v2, msg=message)

  def testReorderBatch(self):
    """Reversing the order of the inputs does not change value of outputs."""
    predictor = self._get_predictor()
    reverse_formulas = self.formulas[::-1]
    reverse_goal_formulas = self.goal_formulas[::-1]
    reverse_embeddings = self.embeddings[::-1]
    self._assert_all_close_batch(
        predictor.batch_goal_embedding(self.goal_formulas)[::-1],
        predictor.batch_goal_embedding(reverse_goal_formulas),
        'Reversing the order of the inputs should not change goal embeddings.')
    self._assert_all_close_batch(
        predictor.batch_thm_embedding(self.formulas)[::-1],
        predictor.batch_thm_embedding(reverse_formulas),
        'Reversing order of the inputs should not change theorem embeddings.')
    self._assert_all_close_batch(
        predictor.batch_tactic_scores(self.embeddings)[::-1],
        predictor.batch_tactic_scores(reverse_embeddings),
        'Reversing the order of the inputs should not change tactic scores.')
    self._assert_all_close_batch(
        predictor.batch_thm_scores(
            self.embeddings[0], self.embeddings,
            tactic_id=self.tactic_id)[::-1],
        predictor.batch_thm_scores(
            self.embeddings[0], reverse_embeddings, tactic_id=self.tactic_id),
        'Reversing the order of the inputs should not change theorem scores.')

  def testBatchThmScores(self):
    """Theorem scores should be based only on goal, thm pair embeddings."""
    predictor = self._get_predictor()
    emb1 = np.tile(1., EMBEDDING_SIZE)
    emb2 = np.tile(0.5, EMBEDDING_SIZE)
    emb3 = np.tile(2., EMBEDDING_SIZE)
    [thm_score] = predictor.batch_thm_scores(emb1, [emb2], tactic_id=0)

    batch_thm_scores = predictor.batch_thm_scores(
        emb1, [emb2, emb3, emb1, emb3], tactic_id=0)

    self.assertAlmostEqual(
        thm_score,
        batch_thm_scores[0],
        places=5,
        msg='Score should be computed pairwise by index over theorems and goal.'
    )
    self.assertRaises(AssertionError, self.assertAlmostEqual,
                      batch_thm_scores[0], batch_thm_scores[1])

  def testGoalEmbedding(self):
    predictor = self._get_predictor()
    goal = '(a (f int bool) (v A y))'
    goal_emb = predictor.goal_embedding(goal)
    [batch_goal_emb] = predictor.batch_goal_embedding([goal])
    self.assertAllEqual(
        goal_emb, batch_goal_emb,
        'Goal embeddings are the same computed in a batch or individually.')

  def testThmEmbedding(self):
    predictor = self._get_predictor()
    thm = '(v A y)'
    thm_emb = predictor.thm_embedding(thm)
    [batch_thm_emb] = predictor.batch_thm_embedding([thm])
    self.assertAllEqual(
        thm_emb, batch_thm_emb,
        'Theorem embeddings are the same computed in a batch or individually.')

  def testProofStateEmbedding(self):
    predictor = self._get_predictor()
    goal = '(a (f bool) (v A y))'
    state = predictions.ProofState(goal=goal)
    goal_emb = predictor.goal_embedding(goal)
    expected_state_emb = predictions.EmbProofState(goal_emb=goal_emb)
    actual_state_emb = predictor.proof_state_embedding(state)
    self.assertAllClose(expected_state_emb.goal_emb, actual_state_emb.goal_emb)
    self.assertAllEqual(expected_state_emb[1:], actual_state_emb[1:])

  def testProofStateEncoding(self):
    predictor = self._get_predictor()
    goal_emb = self.embeddings[0]
    state_emb = predictions.EmbProofState(goal_emb=goal_emb)
    actual_state_enc = predictor.proof_state_encoding(state_emb)
    self.assertAllEqual(goal_emb, actual_state_enc)

  def testGraphReloading(self):
    """Reloading the graph should not affect values."""
    predictor = self._get_predictor()
    new_predictor = self._get_new_predictor()

    self.assertAllEqual(
        predictor.batch_goal_embedding(self.goal_formulas),
        new_predictor.batch_goal_embedding(self.goal_formulas),
        'Reloading the graph should not change goal embeddings.')

    self.assertAllEqual(
        predictor.batch_thm_embedding(self.formulas),
        new_predictor.batch_thm_embedding(self.formulas),
        'Reloading the graph should not change theorem embeddings.')

    self.assertAllEqual(
        predictor.batch_tactic_scores(self.embeddings),
        new_predictor.batch_tactic_scores(self.embeddings),
        'Reloading the graph should not change tactic scores.')

    self.assertAllEqual(
        predictor.batch_thm_scores(
            self.embeddings[0], self.embeddings, tactic_id=self.tactic_id),
        new_predictor.batch_thm_scores(
            self.embeddings[0], self.embeddings, tactic_id=self.tactic_id),
        'Reloading the graph should not change theorem scores.')
