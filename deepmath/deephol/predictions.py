"""An Abstract API for predictions used by the action generator.

Predictions splits the input batches into pieces of max batch size and
concatenates the output arrays if necessary.
"""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function

import abc
import numpy as np

from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Text

# Numpy arrays do not support type checking for arrays with different shapes.
# [goal_emb_size] float32
GOAL_EMB_TYPE = np.ndarray
# [batch_size, Goal_emb_size] float32
BATCH_GOAL_EMB_TYPE = np.ndarray
# [thm_emb_size] float32
THM_EMB_TYPE = np.ndarray
# [batch_size, thm_emb_size] float32
BATCH_THM_EMB_TYPE = np.ndarray
# [state_enc_size] float32
STATE_ENC_TYPE = np.ndarray


def batch_array(array, max_batch_size):
  """Split an array according the maximum batch_size.

  Args:
    array: List or 2D Numpy array.
    max_batch_size: Integer value with maximum batch_size or None.

  Returns:
    A list of lists or numpy arrays the concatenation of which is the input
    list or array, and the first dimension of each array is less than or equal
    to max_batch_size. If max_batch_size is None, then a singleton list with the
    input list/array is returned.
  """
  if max_batch_size is None:
    return [array]
  num_batches = (len(array) + max_batch_size - 1) // max_batch_size
  assert num_batches > 0
  return [
      array[(i * max_batch_size):((i + 1) * max_batch_size)]
      for i in range(num_batches)
  ]


def batched_run(inputs, evaluator, max_batch_size):
  """Run some evaluator function on a set of inputs in a batched manner.

  The input array or list will be chunked into minimum length list of
  batches of size at least max_batch_size, ran through the evaluator and
  the result arrays are concatenated into a final solution. The results are
  assumed to be numpy arrays.

  Args:
    inputs: List of input 1D arrays, strings or dictionaries.
    evaluator: Function to be applied on the produced batches.
    max_batch_size: optional integer, maximum size for the chunks to be
      processed by the evaluator.

  Returns:
    Concatenated result for the batches.
  """
  assert inputs
  # We disable the warning so that this code works with numpy arrays as well.
  if not len(inputs[0]):  # pylint: disable=g-explicit-length-test
    return np.empty([0])
  for i in range(1, len(inputs)):
    assert len(inputs[0]) == len(inputs[i])
  batched_inputs = [batch_array(a, max_batch_size) for a in inputs]
  outputs = [evaluator(*batch) for batch in zip(*batched_inputs)]
  assert outputs
  if len(outputs) == 1:
    return outputs[0]
  else:
    return np.concatenate(outputs)


class ProofState(
    NamedTuple('ProofState', [('goal', Text), ('asl', List[Text]),
                              ('goal_hist', List[Text]), ('orig_conj', Text)])):
  """ProofState contains all values that we want to use for predictions.

  goal: Conclusion term of the goal state.
  asl: List of theorem assumptions for the goal state, typically (h (A) A)
  goal_hist: List of previously visited goals.
  orig_conj: The conclusion term of the original conjecture.
  """
  __slots__ = ()

  def __new__(cls, goal, asl=None, goal_hist=None, orig_conj=None):
    return super(ProofState, cls).__new__(cls, goal, asl, goal_hist, orig_conj)


class EmbProofState(
    NamedTuple('EmbProofState', [('goal_emb', GOAL_EMB_TYPE),
                                 ('asl_emb', BATCH_THM_EMB_TYPE),
                                 ('goal_hist_emb', BATCH_GOAL_EMB_TYPE),
                                 ('orig_conj_emb', GOAL_EMB_TYPE)])):
  """Contains vector embeddings of any strings in ProofState.

  goal_emb: Goal embedding.
  asl_emb: List of assumption embeddings.
  goal_hist_emb: List of goal history embeddings.
  orig_conj_emb: Original conjecture embedding.
  """
  __slots__ = ()

  def __new__(cls,
              goal_emb,
              asl_emb=None,
              goal_hist_emb=None,
              orig_conj_emb=None):
    return super(EmbProofState, cls).__new__(cls, goal_emb, asl_emb,
                                             goal_hist_emb, orig_conj_emb)


class Predictions(object):
  """Compute embeddings and make predictions from a saved checkpoint.

  This class is the abstract base class for all predictions for HOL Light.
  This class uses batches of given maximum size to make the predictions. The
  result is assumed to be numpy arrays of given size and concatenated to be of
  the final size.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self,
               max_embedding_batch_size: Optional[int] = 128,
               max_score_batch_size: Optional[int] = 128) -> None:
    """Restore from the checkpoint into the session."""
    self.max_embedding_batch_size = max_embedding_batch_size
    self.max_score_batch_size = max_score_batch_size

  def goal_embedding(self, goal: Text) -> GOAL_EMB_TYPE:
    """Given a goal as a string, computes and returns its embedding."""
    # Pack and unpack the goal into a batch of size one.
    [embedding] = self.batch_goal_embedding([goal])
    return embedding

  def batch_goal_embedding(self, goals: List[Text]) -> BATCH_GOAL_EMB_TYPE:
    """Computes embeddings from a list of goals."""
    return batched_run([goals], self._batch_goal_embedding,
                       self.max_embedding_batch_size)

  @abc.abstractmethod
  def _batch_goal_embedding(self, goals: List[Text]) -> BATCH_GOAL_EMB_TYPE:
    """Computes embeddings from a list of goals."""
    pass

  def thm_embedding(self, thm: Text) -> THM_EMB_TYPE:
    """Given a theorem as a string, computes and returns its embedding."""
    [embedding] = self.batch_thm_embedding([thm])
    return embedding

  def batch_thm_embedding(self, thms: List[Text]) -> BATCH_THM_EMB_TYPE:
    """From a list of string theorems, computes and returns their embeddings."""
    return batched_run([thms], self._batch_thm_embedding,
                       self.max_embedding_batch_size)

  @abc.abstractmethod
  def _batch_thm_embedding(self, thms: List[Text]) -> BATCH_THM_EMB_TYPE:
    """From a list of string theorems, computes and returns their embeddings."""
    pass

  @abc.abstractmethod
  def proof_state_from_search(self, node) -> ProofState:
    """Convert from proof_search_tree.ProofSearchNode to proof state."""
    pass

  @abc.abstractmethod
  def proof_state_embedding(self, state: ProofState) -> EmbProofState:
    """From a proof state, computes and returns embeddings of each component."""
    pass

  @abc.abstractmethod
  def proof_state_encoding(self, state_emb: EmbProofState) -> STATE_ENC_TYPE:
    """From an embedding of a proof state, computes and returns its encoding."""
    pass

  def batch_tactic_scores(self,
                          state_encodings: List[STATE_ENC_TYPE]) -> np.ndarray:
    """Predicts tactic probabilities for a batch of goals.

    Args:
      state_encodings: A list of n proof state encodings.

    Returns:
      A 2D array [batch_size, num_tactics]. A batch of tactic probabilities.
    """
    return batched_run([state_encodings], self._batch_tactic_scores,
                       self.max_score_batch_size)

  @abc.abstractmethod
  def _batch_tactic_scores(self,
                           state_encodings: List[STATE_ENC_TYPE]) -> np.ndarray:
    """Predicts tactic probabilities for a batch of goals."""
    pass

  def batch_thm_scores(self,
                       state_encoding: STATE_ENC_TYPE,
                       thm_embeddings: BATCH_THM_EMB_TYPE,
                       tactic_id: Optional[int] = None) -> List[float]:
    """Predict relevance scores for goal, theorem pairs.

    Args:
      state_encoding: A proof state encoding.
      thm_embeddings: A list of n theorem embeddings. Theorems are paired by
        index with corresponding goals.
      tactic_id: Optionally tactic that the theorem parameters will be used in.

    Returns:
      A list of n floats, representing the pairwise score of each goal, thm.
    """
    batched_thm_emb = batch_array(thm_embeddings, self.max_score_batch_size)
    if self.max_score_batch_size is None:
      state_copies = np.tile([state_encoding], [len(batched_thm_emb[0]), 1])
    else:
      state_copies = np.tile([state_encoding], [self.max_score_batch_size, 1])
    ret = []
    for thm_emb in batched_thm_emb:
      scores = self._batch_thm_scores(state_copies[:len(thm_emb)], thm_emb,
                                      tactic_id)
      ret.append(scores)
    return np.concatenate(ret)

  @abc.abstractmethod
  def _batch_thm_scores(self,
                        state_encodings: List[STATE_ENC_TYPE],
                        thm_embeddings: BATCH_THM_EMB_TYPE,
                        tactic_id: Optional[int] = None) -> List[float]:
    pass
