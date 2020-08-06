"""An embedding store to make fast prediction of all preceding theorems.

This module contains the class TheoremEmbeddingStore that is used for
storing theorem embeddings and can compute goal parameter scoring for a large
number of theorems.
"""
import os
import re
import time
from typing import List, Optional, Text
import numpy as np
import tensorflow.compat.v1 as tf
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
    normalized_theorems = [
        normalization_lib.normalize(theorem)
        for theorem in theorem_database.theorems
    ]
    self.thm_embeddings = self.predictor.batch_thm_proto_embedding(
        normalized_theorems)

  def compute_embeddings_for_thms_from_db_file(self, file_path: Text) -> None:
    """Compute the embeddings for the theorems given in a test file.

    Args:
      file_path: Path to the text protobuf file containing the theorem database.
    """
    tf.logging.info('Reading theorems database from "%s"', file_path)
    theorem_database = io_util.load_theorem_database_from_file(file_path)
    self.compute_embeddings_for_thms_from_db(theorem_database)

  def read_embeddings(self, pattern: Text) -> None:
    """Read the embeddings and theorem list from the specified pattern.

    Args:
      pattern: Paths in which the embeddings are stored. If it consists of
        multiple files it must be a sharded file of the form *-XXXXX-of-YYYYY
        and all shards must be available.
    """
    tf.logging.info('Reading embeddings from "%s"', pattern)
    paths = tf.gfile.Glob(pattern)
    shards = []
    if not paths:
      raise ValueError('Cannot find embedding store at given pattern: %s' %
                       pattern)
    if len(paths) > 1:
      total_number_of_shards = None
      regex = '-[0-9]*-of-([0-9]*)'
      for path in paths:
        try:
          matches = re.search(regex, path)
          if not matches:
            raise ValueError('Path does not seem to be part of a sharded file: '
                             '%s' % path)
          number_of_shards = int(matches.groups()[0])
        except:  # pylint: disable=broad-except
          raise ValueError('Path needs to be part of a sharded file. %s' % path)
        if total_number_of_shards is None:
          total_number_of_shards = number_of_shards
        if total_number_of_shards != number_of_shards:
          raise ValueError('Inconsistent total number of shards in pattern.')
      if len(paths) != total_number_of_shards:
        raise ValueError('Not all shards of embedding store are available.')

    start_time = time.time()
    for path in sorted(paths):
      with tf.gfile.Open(path, 'rb') as f:
        shard = np.load(f)
        if len(shard) > 0:  # pylint: disable=g-explicit-length-test
          shards.append(shard)
    self.thm_embeddings = np.concatenate(shards)
    tf.logging.info('Read embeddings within %f sec', time.time() - start_time)

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
    thm_embeddings = self.get_embeddings_for_preceding_thms(thm_index)
    assert not self.assumptions
    assert not self.assumption_embeddings
    assert len(thm_embeddings) == thm_index + len(self.assumptions)
    return self.predictor.batch_thm_scores(goal_embedding, thm_embeddings,
                                           tactic_id)
