"""Abstract test suite for instantiations of third_party.deepmath.deephol.predictions."""
import abc
from absl import flags
import numpy as np
import six
import tensorflow.compat.v1 as tf

from deepmath.deephol import predictions
from deepmath.deephol import theorem_utils
from deepmath.proof_assistant import proof_assistant_pb2

FLAGS = flags.FLAGS
NUM_TACTICS = 41
EMBEDDING_SIZE = 4


def _create_theorem_proto(conclusion, tag):
  proto = proof_assistant_pb2.Theorem(
      conclusion=conclusion,
      tag=tag,
      training_split=proof_assistant_pb2.Theorem.TRAINING)
  assumptions_or_hyps = (['(a b)', '(a (b x a) (b x a))', 'hypo', '(a b)']
                         if conclusion else [])
  if tag == proof_assistant_pb2.Theorem.GOAL:
    proto.assumptions.extend(
        map(theorem_utils.assume_term, assumptions_or_hyps))
  else:
    proto.hypotheses.extend(assumptions_or_hyps)
  return proto


class PredictionsAbstractBaseTest(
    six.with_metaclass(abc.ABCMeta, tf.test.TestCase)):

  @classmethod
  @abc.abstractmethod
  def setUpClass(cls):
    """Concrete instances must create cls.predictor."""
    super(PredictionsAbstractBaseTest, cls).setUpClass()

  def setUp(self):
    super(PredictionsAbstractBaseTest, self).setUp()

    self.formulas = [
        '(conclusion1 x)', '(conclusion2 y)', '(a (f bool) (v A y))', '(v A x)',
        '(v A y)', '(a (f int bool) (v A y))', '(= a b)', ''
    ]
    self.embeddings = [
        np.arange(float(i), float(i + EMBEDDING_SIZE))
        for i in range(len(self.formulas))
    ]
    self.tactic_id = 0
    self.theorem_protos = [
        _create_theorem_proto(f, proof_assistant_pb2.Theorem.THEOREM)
        for f in self.formulas
    ]
    self.goal_protos = [
        _create_theorem_proto(f, proof_assistant_pb2.Theorem.GOAL)
        for f in self.formulas
        if f
    ]
    self.proof_state = predictions.ProofState(
        goal=self.goal_protos[1],
        previous_proof_state=predictions.ProofState(goal=self.goal_protos[0]))

  @abc.abstractmethod
  def _get_new_predictor(self, predictor_name):
    pass

  @abc.abstractmethod
  def _get_predictor_map(self):
    pass

  def _testSameEmbeddings(self, formulas):
    """Test that the same formulas are embedded to the same vector."""
    for predictor_name, predictor in self._get_predictor_map().items():
      goal_embs = predictor.batch_goal_embedding(formulas)
      # Goal embeddings must be the same for the same goal string.
      self.assertAllCloseAccordingToType(
          goal_embs[0], goal_embs[1], msg='Failed for %s' % predictor_name)

      # Theorem embeddings must be the same for the same theorem string.
      thm_embs = predictor.batch_thm_embedding(formulas)
      self.assertAllCloseAccordingToType(
          thm_embs[0], thm_embs[1], msg='Failed for %s' % predictor_name)

      # Theorem and goal embeddings should not be the same.
      for ge, te in zip(goal_embs, thm_embs):
        self.assertRaises(
            AssertionError,
            self.assertAllCloseAccordingToType,
            ge,
            te,
            msg='Failed for %s' % predictor_name)

      # Tactic predictor should all be the same.
      tac_pred = predictor.batch_tactic_scores(goal_embs)
      self.assertAllCloseAccordingToType(
          tac_pred[0],
          tac_pred[1],
          msg=('Tactic scores must be the same for the same goal embedding. '
               'Failed for %s' % predictor_name))

      # Tactic predictor should give a probability for each tactic.
      self.assertLen(
          tac_pred[0], NUM_TACTICS,
          'Probability should be predicted for each tactic (%d). Failed for %s'
          % (NUM_TACTICS, predictor_name))
      self.assertLen(
          tac_pred[1], NUM_TACTICS,
          'Probability should be predicted for each tactic (%d). Failed for %s'
          % (NUM_TACTICS, predictor_name))

      self.assertAllCloseAccordingToType(
          tac_pred,
          predictor.batch_tactic_scores(goal_embs),
          msg=('Tactic scores should be the same, even with multiple '
               'executions. Failed for %s' % predictor_name))

  def testSameEmbeddingsExact(self):
    self._testSameEmbeddings(['(v A x)', '(v A x)'])

  def testSameEmbeddingsObscure(self):
    self._testSameEmbeddings([
        '(a not_in_vocab_1 not_in_vocab2)', '(a not_in_vocab_3 not_in_vocab4)'
    ])

  def testDifferentEmbedding(self):
    """Goals are different. Permuting batch or reloading graph has no effect."""
    for predictor_name, predictor in self._get_predictor_map().items():
      goal_embs = predictor.batch_goal_embedding(self.formulas)
      thm_embs = predictor.batch_thm_embedding(self.formulas)

      self.assertLen(
          set([tuple(emb) for emb in goal_embs]), len(self.formulas),
          'Unique goals should be uniquely embedded. Failed for %s' %
          predictor_name)
      self.assertLen(
          set([tuple(emb) for emb in thm_embs]), len(self.formulas),
          'Unique theorems should be uniquely embedded. Failed for %s' %
          predictor_name)

  def _assert_all_close_batch(self,
                              batch1,
                              batch2,
                              message,
                              rtol=1e-06,
                              atol=1e-06):
    for v1, v2 in zip(batch1, batch2):
      self.assertAllClose(v1, v2, msg=message, rtol=rtol, atol=atol)

  def testReorderBatch(self):
    """Reversing the order of the inputs does not change value of outputs."""
    for predictor_name, predictor in self._get_predictor_map().items():
      reverse_formulas = self.formulas[::-1]
      reverse_embeddings = self.embeddings[::-1]
      self._assert_all_close_batch(
          predictor.batch_goal_embedding(self.formulas)[::-1],
          predictor.batch_goal_embedding(reverse_formulas),
          ('Reversing the order of the inputs should not change goal '
           'embeddings. Failed for %s' % predictor_name))
      self._assert_all_close_batch(
          predictor.batch_thm_embedding(self.formulas)[::-1],
          predictor.batch_thm_embedding(reverse_formulas),
          ('Reversing the order of the inputs should not change theorem '
           'embeddings. Failed for %s' % predictor_name))
      self._assert_all_close_batch(
          predictor.batch_tactic_scores(self.embeddings)[::-1],
          predictor.batch_tactic_scores(reverse_embeddings),
          ('Reversing the order of the inputs should not change tactic '
           'scores. Failed for %s' % predictor_name))
      self._assert_all_close_batch(
          predictor.batch_thm_scores(
              self.embeddings[0], self.embeddings,
              tactic_id=self.tactic_id)[::-1],
          predictor.batch_thm_scores(
              self.embeddings[0], reverse_embeddings, tactic_id=self.tactic_id),
          ('Reversing the order of the inputs should not change theorem '
           'scores. Failed for %s' % predictor_name))

  def _test_batch(self, batch, batch_idx, message, precision=1e-06):
    """Adding other values in the batch does not change value of outputs."""
    for predictor_name, predictor in self._get_predictor_map().items():
      self._assert_all_close_batch(
          predictor.batch_goal_embedding(batch)[batch_idx],
          predictor.batch_goal_embedding([batch[batch_idx]])[0],
          message='batch_goal_embedding with predictor <%s>: %s' %
          (predictor_name, message),
          rtol=precision,
          atol=precision)
      self._assert_all_close_batch(
          predictor.batch_thm_embedding(batch)[batch_idx],
          predictor.batch_thm_embedding([batch[batch_idx]])[0],
          message='batch_thm_embedding with predictor <%s>: %s' %
          (predictor_name, message),
          rtol=precision,
          atol=precision)

  def testBatchPartnersLongInputs(self):

    def generate_long_input(depth):
      if depth <= 0:
        return 'x'
      else:
        return '(v A %s)' % generate_long_input(depth - 1)

    self._test_batch(
        ['(v A x)', generate_long_input(900)],
        0,
        'Adding long inputs to the batch should not change the embedding.',
        precision=1e-04)

  def testBatchPartnersBigBatch(self):
    batch_size = 512
    big_batch = ['(v A x)' for _ in range(batch_size - 1)]
    formula_idx = int(batch_size / 2)
    big_batch.insert(formula_idx, '(v A y)')
    self._test_batch(big_batch, formula_idx,
                     'Very large batches should not change the embedding.')

  def testBatchThmScores(self):
    """Theorem scores should be based only on goal, thm pair embeddings."""
    for predictor_name, predictor in self._get_predictor_map().items():
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
          msg=('Score should be computed pairwise by index over theorems and '
               'goal. Failed for %s' % predictor_name))
      self.assertRaises(AssertionError, self.assertAlmostEqual,
                        batch_thm_scores[0], batch_thm_scores[1])

  def testGoalEmbedding(self):
    for predictor_name, predictor in self._get_predictor_map().items():
      for goal in self.formulas:
        goal_emb = predictor.goal_embedding(goal)
        [batch_goal_emb] = predictor.batch_goal_embedding([goal])
        self.assertAllEqual(
            goal_emb, batch_goal_emb,
            'Goal embeddings are the same computed in a batch or individually.'
            'Failed for %s on goal: %s' % (predictor_name, goal))

  def testGoalProtoEmbedding(self):
    for predictor_name, predictor in self._get_predictor_map().items():
      for goal in self.goal_protos:
        goal_emb = predictor.goal_proto_embedding(goal)
        [batch_goal_emb] = predictor.batch_goal_proto_embedding([goal])
        self.assertAllEqual(
            goal_emb, batch_goal_emb,
            'Goal proto embeddings are the same computed in a batch or '
            'individually. Failed for %s on goal:\n%s' % (predictor_name, goal))

  def testThmEmbedding(self):
    for predictor_name, predictor in self._get_predictor_map().items():
      for thm in self.formulas:
        thm_emb = predictor.thm_embedding(thm)
        [batch_thm_emb] = predictor.batch_thm_embedding([thm])
        self.assertAllEqual(
            thm_emb, batch_thm_emb,
            'Thm embeddings are the same computed in a batch or individually.'
            'Failed for %s on theorem: %s' % (predictor_name, thm))

  def testTheoremProtoEmbedding(self):
    for predictor_name, predictor in self._get_predictor_map().items():
      for theorem in self.theorem_protos:
        theorem_emb = predictor.thm_proto_embedding(theorem)
        [batch_theorem_emb] = predictor.batch_thm_proto_embedding([theorem])
        self.assertAllEqual(
            theorem_emb, batch_theorem_emb,
            'Theorem proto embeddings are the same computed in a batch or '
            'individually. Failed for %s on theorem:\n%s' %
            (predictor_name, theorem))

  def testProofStateEncoding(self):
    for predictor_name, predictor in self._get_predictor_map().items():
      for idx, goal_emb in enumerate(self.embeddings):
        state_emb = predictions.EmbProofState(goal_emb=goal_emb)
        actual_state_enc = predictor.proof_state_encoding(state_emb)
        self.assertAllEqual(
            goal_emb,
            actual_state_enc,
            msg=('Proof state encoding just returns the goal embedding.'
                 ' Failed for %s on embeddings[%d]' % (predictor_name, idx)))

  def testSearchStateScore(self):
    for predictor_name, predictor in self._get_predictor_map().items():

      predicted_score = predictor.search_state_score(
          predictions.ProofState(search_state=self.goal_protos))

      self.assertBetween(predicted_score, 0., 1.,
                         ('Predicted search state score must be in [0, 1]. '
                          'Failed for %s' % predictor_name))

      self.assertEqual(
          predicted_score,
          predictor.search_state_score(
              predictions.ProofState(search_state=self.goal_protos)),
          ('Search state score should be the same for the same proof states. '
           'Failed for %s' % predictor_name))

      self.assertNotEqual(
          predictor.search_state_score(
              predictions.ProofState(search_state=self.goal_protos[:1])),
          predictor.search_state_score(
              predictions.ProofState(search_state=self.goal_protos[2:])),
          ('Search state score should be different for different proof states. '
           'Failed for %s' % predictor_name))

  def testGraphReloading(self):
    """Reloading the graph should not affect values."""
    for predictor_name, predictor in self._get_predictor_map().items():
      new_predictor = self._get_new_predictor(predictor_name)

      self.assertAllEqual(
          predictor.batch_goal_embedding(self.formulas),
          new_predictor.batch_goal_embedding(self.formulas),
          ('Reloading the graph should not change goal embeddings. '
           'Failed for %s' % predictor_name))

      self.assertAllEqual(
          predictor.batch_thm_embedding(self.formulas),
          new_predictor.batch_thm_embedding(self.formulas),
          ('Reloading the graph should not change theorem embeddings. '
           'Failed for %s' % predictor_name))

      self.assertAllEqual(
          predictor.batch_tactic_scores(self.embeddings),
          new_predictor.batch_tactic_scores(self.embeddings),
          ('Reloading the graph should not change tactic scores. '
           'Failed for %s' % predictor_name))

      self.assertAllEqual(
          predictor.batch_thm_scores(
              self.embeddings[0], self.embeddings, tactic_id=self.tactic_id),
          new_predictor.batch_thm_scores(
              self.embeddings[0], self.embeddings, tactic_id=self.tactic_id),
          ('Reloading the graph should not change theorem scores. '
           'Failed for %s' % predictor_name))

      self.assertAllEqual(
          predictor.search_state_score(
              predictions.ProofState(search_state=self.goal_protos)),
          new_predictor.search_state_score(
              predictions.ProofState(search_state=self.goal_protos)),
          ('Reloading the graph should not change the search state scores. '
           'Failed for %s' % predictor_name))


if __name__ == '__main__':
  tf.test.main()
