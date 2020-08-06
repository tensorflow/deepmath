"""An Abstract API for predictions used by the action generator.

Predictions splits the input batches into pieces of max batch size and
concatenates the output arrays if necessary.
"""
import abc
import os
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Text
import numpy as np
import tensorflow.compat.v1 as tf
from deepmath.proof_assistant import proof_assistant_pb2

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
    NamedTuple('ProofState',
               [('goal', Optional[proof_assistant_pb2.Theorem]),
                ('targets', Optional[List[proof_assistant_pb2.Theorem]]),
                ('search_state', Optional[List[proof_assistant_pb2.Theorem]]),
                ('previous_proof_state', Optional['ProofState'])])):
  """Contains all information that we use for predictions.

  goal: A single explicitly designated current goal. We select a parametrized
  tactic to apply on this goal.
  search state: A list of goals in the proof search tree. Closing all
  of these goals would close the original root goal.
  previous_proof_state: A reference to the previous proof state that has led to
  this proof state.

  Example:
                    a
  tactic -----------|----------- tactic
         |                     |
         |              -------|-----------
         |              |                 |
         b              c              ---d---
                 tactic |       tactic |     | tactic
                        |              |     |
                     ---|---           |     |
                     |     |           |     |
                     e     f           g     h

  In this proof search tree, consider state1 with e as the current goal and
  [e, f, h] as the search state. The previous proof state of state1 is state2
  with c as the current goal and [c, h] as the search state. The previous proof
  state of state2 is state3 with a as the current goal, [a] as the search state,
  and no previous proof state (ie this is the first proof state).
  """
  __slots__ = ()

  def __new__(cls,
              goal=None,
              targets=None,
              search_state=None,
              previous_proof_state=None):
    return super(ProofState, cls).__new__(cls, goal, targets, search_state,
                                          previous_proof_state)


class EmbProofState(NamedTuple('EmbProofState', [('goal_emb', GOAL_EMB_TYPE)])):
  """Contains embeddings of fields in ProofState.

  goal_emb: Goal embedding.
  """
  __slots__ = ()

  def __new__(cls, goal_emb):
    return super(EmbProofState, cls).__new__(cls, goal_emb)


def get_saved_model_path(training_ckpt_base):
  """Return the path to the eval graph, if it exists.

  Args:
    training_ckpt_base: String representing the checkpoint base, e.g.
      model.ckpt-0

  Returns:
    The path to a saved model protobuff or None if none is found.
  """
  ckpt_dir = os.path.dirname(training_ckpt_base)
  # If using a checkpoint from the best_exporter, return its saved_model.
  if os.path.basename(ckpt_dir) == 'variables':
    return os.path.join(
        os.path.dirname(ckpt_dir),
        tf.saved_model.constants.SAVED_MODEL_FILENAME_PB)
  # If using a training checkpoint, still return the eval saved_model.
  else:
    saved_models_dir = os.path.join(ckpt_dir, 'export', 'best_exporter')
    saved_model_paths = tf.gfile.Glob(os.path.join(saved_models_dir, '*'))
    if saved_model_paths:
      return os.path.join(saved_model_paths[0],
                          tf.saved_model.constants.SAVED_MODEL_FILENAME_PB)
    # Otherwise, there is not eval saved_model.
    else:
      return None


class Predictions(object):  # pytype: disable=ignored-metaclass
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

  def goal_proto_embedding(self,
                           goal: proof_assistant_pb2.Theorem) -> GOAL_EMB_TYPE:
    """Given a Goal proto, computes and returns its embedding."""
    [embedding] = self.batch_goal_proto_embedding([goal])
    return embedding

  @abc.abstractmethod
  def batch_goal_proto_embedding(
      self, goals: List[proof_assistant_pb2.Theorem]) -> BATCH_GOAL_EMB_TYPE:
    """From a list of Goal protos, computes and returns their embeddings."""
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

  def thm_proto_embedding(self,
                          theorem: proof_assistant_pb2.Theorem) -> THM_EMB_TYPE:
    """Given a Theorem proto, computes and returns its embedding."""
    [embedding] = self.batch_thm_proto_embedding([theorem])
    return embedding

  @abc.abstractmethod
  def batch_thm_proto_embedding(
      self, theorems: List[proof_assistant_pb2.Theorem]) -> BATCH_THM_EMB_TYPE:
    """From a list of Theorem protos, computes and returns their embeddings."""
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

  def search_state_score(self, proof_state: ProofState) -> float:
    """Basic search state estimated values based on length.

    This function allows MCTS prover without a neural network. However, it
    should be overwritten with a function that uses a neural network estimator.

    Args:
      proof_state: Representation of the state to evaluate.

    Returns:
      Value in [0, 1], where 1 is likely provable and 0 is likely not proveable.
    """
    tf.logging.warn('Using built-in MCTS search state scorer. '
                    'Please replace with trained predictor.')

    goals_length = 0
    for goal in proof_state.search_state:
      goals_length += len(goal.conclusion)
      for hyp in goal.hypotheses:
        goals_length += len(hyp)
    return min(0.1, 1 / (goals_length + 1.))
