"""Base class for action generator.

Abstract base class for action generators.
"""

import abc
import collections
import time
from typing import List

import tensorflow.compat.v1 as tf

from deepmath.deephol import deephol_pb2
from deepmath.deephol import proof_search_tree
from deepmath.deephol import theorem_fingerprint
from deepmath.proof_assistant import proof_assistant_pb2

Suggestion = collections.namedtuple('Suggestion', ['tactic', 'params', 'score'])


class AbstractActionGenerator:  # pytype: disable=ignored-metaclass
  """Abstract class for action generators."""

  __metaclass__ = abc.ABCMeta

  def __init__(
      self,
      theorem_database: proof_assistant_pb2.TheoremDatabase,
      tactics: List[deephol_pb2.Tactic],
  ):
    for idx, tactic in enumerate(tactics):
      if tactic.id != idx:
        raise ValueError('Assumed tactics to arrive in order. '
                         'Found tactic %s at position %d.' % (str(tactic), idx))
    self.theorem_database = theorem_database
    self.tactics = tactics
    self.thm_index_by_fingerprint = {
        theorem_fingerprint.Fingerprint(thm): i
        for (i, thm) in enumerate(theorem_database.theorems)
    }

  def step(self, node: proof_search_tree.ProofSearchNode,
           premises: proof_assistant_pb2.PremiseSet) -> List[Suggestion]:
    """Generates a list of possible ApplyTactic argument strings from a goal.

    Args:
      node: state of the proof search, starting at current goal.
      premises: Specification of the selection of premises that can be used for
        tactic parameters. Currently we are supporting only a single
        DatabaseSection.

    Returns:
      List of string arguments for HolLight.ApplyTactic function, along with
      scores (Suggestion).
    """
    assert not premises.reference_sets, ('Premise reference sets are not '
                                         'supported.')
    assert len(premises.sections) == 1, ('Premise set must have exactly one '
                                         'section.')
    # TODO(szegedy): If the premise is not specified, we want the whole
    # database to be used. Not sure if -1 or len(database.theorems) would do
    # that or not. Assertion will certainly fail before that.
    # Also we don't have checks on this use case.
    assert premises.sections[0].HasField('before_premise'), ('Premise is '
                                                             'required.')
    fp = premises.sections[0].before_premise
    thm_number = self.thm_index_by_fingerprint.get(fp)
    assert thm_number is not None
    assert theorem_fingerprint.Fingerprint(
        self.theorem_database.theorems[thm_number]) == fp
    action_gen_start_time = time.time()
    suggestions = self._step(node, thm_number)
    tf.logging.info('action generator suggestions: %s', str(suggestions))
    node.action_generation_time_millisec = int(
        round(1000.0 * (time.time() - action_gen_start_time)))
    return suggestions

  @abc.abstractmethod
  def _step(self, node: proof_search_tree.ProofSearchNode,
            thm_number: int) -> List[Suggestion]:
    pass
