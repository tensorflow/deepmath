"""An embedding store to make fast prediction of all preceding theorems.

This module contains the class TheoremEmbeddingStore that is used for
storing theorem embeddings and can compute goal parameter scoring for a large
number of theorems.
"""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from typing import List, Optional, Text
from deepmath.deephol import io_util
from deepmath.deephol import predictions
from deepmath.deephol.utilities import normalization_lib
from deepmath.proof_assistant import proof_assistant_pb2


class TheoremEmbeddingStore(object):
  """An embedding stores computes and stores embeddings in the given order.

  Either compute_embeddings or read_embeddings should be called before
  save_embeddings or get_thm_score_for_preceding_thms are called.
  """

  def __init__(self, predictor: predictions.Predictions) -> None:
    """Initialize the prediction lib.

    Stores the prediction objects and initializes with an empty store.

    Args:
      predictor: An object conforming to the interface of predictor.Predictions.
    """
    self.predictor = predictor
    self.thm_embeddings = None
    self.assumptions = []
    self.assumption_embeddings = None

  def compute_assumption_embeddings(self, assumptions: List[Text]) -> None:
    """DEPRECATED - Compute embeddings for a list of assumptions and store them.

    The assumptions are preprocessed by truncting by the truncation value
    specified in the constructor.

    Args:
      assumptions: List of assumptions. Their order will be preserved.
    """
    raise NotImplementedError(
        'Computing embedding of assumptions is not implemented.')

  def compute_embeddings_for_thms_from_db(
      self, theorem_database: proof_assistant_pb2.TheoremDatabase) -> None:
    normalized_thms = [
        normalization_lib.normalize(thm).conclusion
        for thm in theorem_database.theorems
    ]
    self.thm_embeddings = self.predictor.batch_thm_embedding(normalized_thms)

  def compute_embeddings_for_thms_from_db_file(self, file_path: Text) -> None:
    """Compute the embeddings for the theorems given in a test file.

    Args:
      file_path: Path to the text protobuf file containing the theorem database.
    """
    tf.logging.info('Reading theorems database from "%s"', file_path)
    theorem_database = io_util.load_theorem_database_from_file(file_path)
    self.compute_embeddings_for_thms_from_db(theorem_database)

  def read_embeddings(self, file_path: Text) -> None:
    """Read the embeddings and theorem list from the specified files.

    Args:
      file_path: Path to the file in which the embeddings are stored.
    """
    tf.logging.info('Reading embeddings from "%s"', file_path)
    with tf.gfile.Open(file_path, 'rb') as f:
      self.thm_embeddings = np.load(f)

  def save_embeddings(self, file_path: Text):
    """Save the embeddings and theorem list to the specified files.

    Args:
      file_path: The name of the file path in which the embeddings are stored.
        The directory and all parent directories are created if necessary.
    """
    dir_name = os.path.dirname(file_path)
    tf.logging.info('Writing embeddings "%s"', file_path)
    if not tf.gfile.Exists(dir_name):
      tf.gfile.MakeDirs(dir_name)
      assert tf.gfile.Exists(dir_name)
    with tf.gfile.Open(file_path, 'wb') as f:
      np.save(f, self.thm_embeddings)

  def get_embeddings_for_preceding_thms(self, thm_index):
    assert thm_index <= self.thm_embeddings.shape[0]
    assert thm_index >= 0
    return self.thm_embeddings[:thm_index]

  def get_thm_scores_for_preceding_thms(self,
                                        goal_embedding,
                                        thm_index: Optional[int] = None,
                                        tactic_id: Optional[int] = None):
    """Get the predicted pairwise scores in a numpy array.

    For the given goal embedding (which is either the embedding of the goal term
    or the embedding of the current proof state), get all the theorem scores
    that preceed the given theorem in theorem list and all the local
    assumptions stored in this store. The theorem parameter thm must be either
    None or be in the theorem list, otherwise an assertion will fail.

    Args:
      goal_embedding: 1D embedding with the embedding of the given goal.
      thm_index: Theorem index in the list of theorems in this store or None, in
        which case all of the theorems are scored.
      tactic_id: Optionally tactic that the theorem parameters will be used in.

    Returns:
      A 1D numpy array with the same length as the sum of the length
      of preceding thms and assumptions. It is the concatenated array of the
      scores for the preceding thms and assumptions in the same order given as
      in the those arrays: first the theorem scores, then the assumption scores.
    """
    if thm_index is None:
      thm_index = self.thm_embeddings.shape[0]
    else:
      assert thm_index <= self.thm_embeddings.shape[0]
      assert thm_index >= 0
    assert not self.assumptions
    assert not self.assumption_embeddings
    thm_embeddings = self.thm_embeddings[:thm_index]
    assert len(thm_embeddings) == thm_index + len(self.assumptions)
    return self.predictor.batch_thm_scores(goal_embedding, thm_embeddings,
                                           tactic_id)
