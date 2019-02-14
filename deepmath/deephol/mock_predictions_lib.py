"""Test Mock class for predictions.Predictions.

This class can be used for mocking concrete Predictions objects.
"""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function

import numpy as np

from deepmath.deephol import predictions


# TODO(smloos): Rename to MockPredictions.
class MockPredictionsLib(predictions.Predictions):
  """Mock Class for predictions.Predictions."""

  def _batch_goal_embedding(self, goals):
    return np.array([[goal.__hash__(), 0] for goal in goals])

  def _batch_thm_embedding(self, thms):
    """From a list of string theorems, compute and return their embeddings."""
    return np.array([[thm.__hash__(), 1] for thm in thms])

  def proof_state_from_search(self, node):
    return predictions.ProofState(goal='goal')

  def proof_state_embedding(self, state: predictions.ProofState):
    return predictions.EmbProofState(*[[x.__hash__(), 2] for x in state])

  def proof_state_encoding(self, state_emb: predictions.EmbProofState):
    return np.array([state_emb.__hash__(), 3])

  def _batch_tactic_scores(self, goal_embeddings):
    return np.array([[np.sum(emb), 0.0] for emb in goal_embeddings])

  def _batch_thm_scores(self, goal_embeddings, thm_embeddings, tactic_id=None):
    if tactic_id is not None:
      c = 3.0 * float(tactic_id)
    else:
      c = 2.0
    return np.array([
        np.sum(e1) + c * np.sum(e2)
        for (e1, e2) in zip(goal_embeddings, thm_embeddings)
    ])


MOCK_PREDICTOR = MockPredictionsLib()
