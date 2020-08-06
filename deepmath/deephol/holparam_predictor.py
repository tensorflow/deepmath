"""Compute embeddings and predictions from a saved holparam checkpoint."""
from typing import List, Optional, Text
import numpy as np
import tensorflow.compat.v1 as tf
from deepmath.deephol import predictions
from deepmath.deephol import process_sexp
from deepmath.proof_assistant import proof_assistant_pb2
from tensorflow.core.protobuf import saved_model_pb2

TRUNCATE_SEXPRESSION = 1000
GOAL_EMB_TYPE = predictions.GOAL_EMB_TYPE
BATCH_GOAL_EMB_TYPE = predictions.BATCH_GOAL_EMB_TYPE
THM_EMB_TYPE = predictions.THM_EMB_TYPE
STATE_ENC_TYPE = predictions.STATE_ENC_TYPE
BATCH_THM_EMB_TYPE = predictions.BATCH_THM_EMB_TYPE


class HolparamPredictor(predictions.Predictions):
  """Compute embeddings and make predictions from a save checkpoint."""

  def __init__(self,
               ckpt: Text,
               max_embedding_batch_size: Optional[int] = 128,
               max_score_batch_size: Optional[int] = 128) -> None:
    """Restore from the checkpoint into the session."""
    super(HolparamPredictor, self).__init__(
        max_embedding_batch_size=max_embedding_batch_size,
        max_score_batch_size=max_score_batch_size)
    self._graph = tf.Graph()
    self._sess = tf.Session(graph=self._graph)
    with self._graph.as_default():
      saved_model_path = predictions.get_saved_model_path(ckpt)
      if saved_model_path:
        self._training_meta = False
        tf.logging.info('Importing from metagraph in %s.', saved_model_path)
        saved_model = saved_model_pb2.SavedModel()
        with tf.gfile.GFile(saved_model_path, 'rb') as f:
          saved_model.ParseFromString(f.read())
          metagraph = saved_model.meta_graphs[0]
      else:
        self._training_meta = True
        metagraph = ckpt + '.meta'
        tf.logging.warn(
            'No exported eval graph found. Using training metagraph %s',
            metagraph)
      # Load metagraph from proto or filepath
      saver = tf.train.import_meta_graph(metagraph, clear_devices=True)
      saver.restore(self._sess, ckpt)
      table_init_op = tf.tables_initializer()
      self._sess.run(table_init_op)
      self.pairwise_score = tf.squeeze(
          self._graph.get_collection('pairwise_score'), axis=[2])

  def __del__(self) -> None:
    """Close the session when deleted."""
    self._sess.close()

  def _goal_string_for_predictions(self, goals: List[Text]) -> List[Text]:
    return [
        process_sexp.process_sexp(goal, truncate=TRUNCATE_SEXPRESSION)
        for goal in goals
    ]

  def _thm_string_for_predictions(self, thms: List[Text]) -> List[Text]:
    return [process_sexp.process_sexp(thm, truncate=1000) for thm in thms]

  def _batch_goal_embedding(self, goals: List[Text]) -> List[GOAL_EMB_TYPE]:
    """From a list of string goals, compute and return their embeddings."""
    # Get the first goal_net collection (second entry may be duplicated to align
    # with negative theorems)
    goals = self._goal_string_for_predictions(goals)
    embeddings = self._sess.run(
        fetches=self._graph.get_collection('goal_net'),
        feed_dict={self._graph.get_collection('goal_string')[0]: goals})[0]
    return embeddings

  def batch_goal_proto_embedding(
      self, goals: List[proof_assistant_pb2.Theorem]) -> BATCH_GOAL_EMB_TYPE:
    """From a list of Goal protos, computes and returns their embeddings."""
    return self.batch_goal_embedding([goal.conclusion for goal in goals])

  def _batch_thm_embedding(self, thms: List[Text]) -> List[THM_EMB_TYPE]:
    """From a list of string theorems, compute and return their embeddings."""
    # The checkpoint should have exactly one value in this collection.
    thms = self._thm_string_for_predictions(thms)
    embeddings = self._sess.run(
        fetches=self._graph.get_collection('thm_net'),
        feed_dict={self._graph.get_collection('thm_string')[0]: thms})[0]
    return embeddings

  def thm_embedding(self, thm: Text) -> THM_EMB_TYPE:
    """Given a theorem as a string, compute and return its embedding."""
    # Pack and unpack the thm into a batch of size one.
    [embedding] = self.batch_thm_embedding([thm])
    return embedding

  def batch_thm_proto_embedding(
      self, theorems: List[proof_assistant_pb2.Theorem]) -> BATCH_THM_EMB_TYPE:
    """From a list of Theorem protos, computes and returns their embeddings."""
    return self.batch_thm_embedding(
        [theorem.conclusion for theorem in theorems])

  def proof_state_embedding(
      self, state: predictions.ProofState) -> predictions.EmbProofState:
    return predictions.EmbProofState(
        goal_emb=self.goal_embedding(state.goal.conclusion))

  def proof_state_encoding(self,
                           state: predictions.EmbProofState) -> STATE_ENC_TYPE:
    return state.goal_emb

  def _batch_tactic_scores(
      self, state_encodings: List[STATE_ENC_TYPE]) -> List[List[float]]:
    """Predict tactic probabilities for a batch of goals.

    Args:
      state_encodings: A list of n goal embeddings.

    Returns:
      A list of n tactic probabilities, each of length equal to the number of
        tactics.
    """
    # The checkpoint should have only one value in this collection.
    feed_dict = {self._graph.get_collection('tactic_net')[0]: state_encodings}
    if self._training_meta:
      feed_dict[self._graph.get_collection('tac_keep_prob')[0]] = 1.0
    [tactic_scores] = self._sess.run(
        fetches=self._graph.get_collection('tactic_logits'),
        feed_dict=feed_dict)
    return tactic_scores

  def _batch_thm_scores(self,
                        state_encodings: List[STATE_ENC_TYPE],
                        thm_embeddings: List[THM_EMB_TYPE],
                        tactic_id: Optional[int] = None) -> List[float]:
    """Predict relevance scores for goal, theorem pairs.

    Args:
      state_encodings: A proof state encoding.
      thm_embeddings: A list of n theorem embeddings. Theorems are paired by
        index with corresponding goals.
      tactic_id: Optionally tactic that the theorem parameters will be used in.

    Returns:
      A list of n floats, representing the pairwise score of each goal, thm.
    """
    del tactic_id  # tactic id not use to predict theorem scores.
    # The checkpoint should have only one value in this collection.
    feed_dict = {
        self._graph.get_collection('goal_net')[-1]: state_encodings,
        self._graph.get_collection('thm_net')[-1]: thm_embeddings
    }
    if self._training_meta:
      feed_dict[self._graph.get_collection('fc_keep_prob')[0]] = 1.0
    [scores] = self._sess.run(fetches=self.pairwise_score, feed_dict=feed_dict)
    return scores


class TacDependentPredictor(HolparamPredictor):
  """Derived class, adds dependence on tactic for computing theorem scores."""

  def __init__(self,
               ckpt: Text,
               max_embedding_batch_size: Optional[int] = 128,
               max_score_batch_size: Optional[int] = 128) -> None:
    """Restore from the checkpoint into the session."""
    super(TacDependentPredictor, self).__init__(
        ckpt,
        max_embedding_batch_size=max_embedding_batch_size,
        max_score_batch_size=max_score_batch_size)
    self.selected_tactic = -1

  def _batch_thm_scores(self,
                        state_encodings: List[STATE_ENC_TYPE],
                        thm_embeddings: List[THM_EMB_TYPE],
                        tactic_id: Optional[int] = None) -> List[float]:
    """Predict relevance scores for goal, theorem pairs.

    Args:
      state_encodings: A proof state encoding.
      thm_embeddings: A list of n theorem embeddings. Theorems are paired by
        index with corresponding goals.
      tactic_id: Optionally tactic that the theorem parameters will be used in.

    Returns:
      A list of n floats, representing the pairwise score of each goal, thm.
    """
    # Check that the batch size for states and thms is the same.
    assert len(state_encodings) == len(thm_embeddings)
    # Tile the tactic to the batch size.
    tactic_ids = np.tile(tactic_id, [len(state_encodings)])
    # The checkpoint should have only one value in this collection.
    feed_dict = {
        self._graph.get_collection('goal_net')[-1]: state_encodings,
        self._graph.get_collection('thm_net')[-1]: thm_embeddings,
        self._graph.get_collection('label_tac_id')[0]: tactic_ids
    }
    if self._training_meta:
      feed_dict[self._graph.get_collection('fc_keep_prob')[0]] = 1.0
    [scores] = self._sess.run(fetches=self.pairwise_score, feed_dict=feed_dict)
    return scores
