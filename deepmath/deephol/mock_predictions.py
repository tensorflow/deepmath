"""Test Mock class for predictions.Predictions.

This class can be used for mocking concrete Predictions objects.
"""
from typing import List, Optional, Text
import numpy as np
from deepmath.deephol import predictions
from deepmath.proof_assistant import proof_assistant_pb2

GOAL_EMB_TYPE = predictions.GOAL_EMB_TYPE
BATCH_GOAL_EMB_TYPE = predictions.BATCH_GOAL_EMB_TYPE
STATE_ENC_TYPE = predictions.STATE_ENC_TYPE
BATCH_THM_EMB_TYPE = predictions.BATCH_THM_EMB_TYPE


class MockPredictions(predictions.Predictions):
  """Mock Class for predictions.Predictions."""

  def _batch_goal_embedding(self, goals: List[Text]) -> List[GOAL_EMB_TYPE]:
    return [np.array([goal.__hash__(), 0]) for goal in goals]

  def batch_goal_proto_embedding(
      self, goals: List[proof_assistant_pb2.Theorem]) -> BATCH_GOAL_EMB_TYPE:
    """From a list of Goal protos, computes and returns their embeddings."""
    def goal_hash(goal):
      goal_conclusion_hash = goal.conclusion.__hash__()
      assumptions_hash = 0
      for hyp in goal.hypotheses:
        assumptions_hash ^= hyp.__hash__()
      for assumption in goal.assumptions:
        assumptions_hash ^= assumption.conclusion.__hash__()
        for hyp in assumption.hypotheses:
          assumptions_hash ^= hyp.__hash__()
      return [goal_conclusion_hash, assumptions_hash]

    return np.array([goal_hash(goal) for goal in goals])

  def _batch_thm_embedding(self, thms: List[Text]) -> BATCH_THM_EMB_TYPE:
    """From a list of string theorems, compute and return their embeddings."""
    return np.array([[thm.__hash__(), 1] for thm in thms])

  def batch_thm_proto_embedding(
      self, thms: List[proof_assistant_pb2.Theorem]) -> BATCH_THM_EMB_TYPE:
    """From a list of Theorem protos, computes and returns their embeddings."""
    def theorem_hash(theorem):
      theorem_conclusion_hash = theorem.conclusion.__hash__()
      hypotheses_hash = 1
      for hypothesis in theorem.hypotheses:
        hypotheses_hash ^= hypothesis.__hash__()
      return [theorem_conclusion_hash, hypotheses_hash]

    return np.array([theorem_hash(thm) for thm in thms])

  def proof_state_embedding(
      self, state: predictions.ProofState) -> predictions.EmbProofState:
    return predictions.EmbProofState(*[[x.__hash__(), 2] for x in state])

  def proof_state_encoding(
      self, state_emb: predictions.EmbProofState) -> STATE_ENC_TYPE:
    return np.array([state_emb.__hash__(), 3])

  def _batch_tactic_scores(self,
                           state_encodings: List[STATE_ENC_TYPE]) -> np.ndarray:
    return np.array([[np.sum(enc), 0.0] for enc in state_encodings])

  def _batch_thm_scores(self,
                        state_encodings: List[STATE_ENC_TYPE],
                        thm_embeddings: BATCH_THM_EMB_TYPE,
                        tactic_id: Optional[int] = None) -> List[float]:
    if tactic_id is not None:
      c = 3.0 * float(tactic_id)
    else:
      c = 2.0
    return [
        np.sum(e1) + c * np.sum(e2)
        for (e1, e2) in zip(state_encodings, thm_embeddings)
    ]

  def search_state_score(self, proof_state: predictions.ProofState) -> float:
    return 0.


MOCK_PREDICTOR = MockPredictions()
