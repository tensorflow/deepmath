"""Proof pruning library.

The purpose of this library is to optimize proofs. Currently we
minimize the number of tactic application parameters in oder to generate
better training data (with minimum number of tactic parameters).
"""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function
import time
import tensorflow as tf
from typing import List, Text
from deepmath.deephol.public import proof_assistant
from deepmath.deephol import deephol_pb2
from deepmath.deephol import prover_util
from deepmath.proof_assistant import proof_assistant_pb2
from deepmath.public import error

MIN_HARD_NEGATIVES = 5
MAX_HARD_NEGATIVES = 10


def _create_request(goal: proof_assistant_pb2.Theorem, tactic: Text,
                    params: List[proof_assistant_pb2.Theorem]
                   ) -> proof_assistant_pb2.ApplyTacticRequest:
  tactic = ('%s [ %s ]' % (tactic, ' ; '.join(
      ['THM %d' % thm.fingerprint for thm in params]))).replace('  ', ' ')
  return proof_assistant_pb2.ApplyTacticRequest(
      goal=prover_util.theorem_to_goal_proto(goal), tactic=tactic)


def _matches_subgoal(goal: proof_assistant_pb2.Theorem,
                     thm: proof_assistant_pb2.Theorem):
  return (set(list(goal.hypotheses)) == set(list(thm.hypotheses)) and
          goal.conclusion == thm.conclusion)


class ParameterPruning(object):
  """Class to do parameter pruning on proof nodes."""

  def __init__(self,
               theorem_db: proof_assistant_pb2.TheoremDatabase,
               hol_wrapper=None):
    if hol_wrapper and theorem_db:
      tf.logging.warning(
          'theorem_db provided will be ignored as hol_wrapper provided.')
    self.hol_wrapper = hol_wrapper
    if not self.hol_wrapper:
      self.hol_wrapper = proof_assistant.ProofAssistant()
      for theorem in theorem_db.theorems:
        self.hol_wrapper.RegisterTheorem(
            proof_assistant_pb2.RegisterTheoremRequest(theorem=theorem))
    self.communication_failed = False

  def prune_tactic_application(self, goal: proof_assistant_pb2.Theorem,
                               tapp: deephol_pb2.TacticApplication):
    """Parameter pruning for a single tactic application.

    Args:
      goal: Goal of the ProofNode to which the tactic application belongs.
      tapp: The tactic application to be pruned.
    """
    if self.communication_failed:
      tf.logging.error('Communication with prover failed. Not pruning...')
      return
    tactic = tapp.tactic
    parameters = tapp.parameters
    if not parameters:
      return
    assert len(parameters) == 1
    param = parameters[0]
    if param.parameter_type != deephol_pb2.Tactic.THEOREM_LIST:
      return
    thms = list(param.theorems)
    if not thms:
      return
    index = len(thms) - 1
    tactic = tapp.tactic
    time_spent = tapp.time_spent
    false_positives = []
    other_negatives = []
    found_true_positive = False
    while index >= 0:
      thm = thms.pop(index)
      request = _create_request(goal, str(tactic), thms)
      start_time = time.time()
      response = proof_assistant_pb2.ApplyTacticResponse()
      try:
        response = self.hol_wrapper.ApplyTactic(request)
        elapsed_msecs = int((time.time() - start_time) * 1000.0 + 0.5)
        time_spent = elapsed_msecs
      except error.StatusNotOk as exception:
        tf.logging.error(exception)
        elapsed_msecs = int((time.time() - start_time) * 1000.0 + 0.5)
        if exception.message.startswith(
            'Communication') and exception.message.endswith('failed.'):
          tf.logging.error('Communication with prover failed. Not pruning...')
          self.communication_failed = True
          return
      if response.HasField('error'):
        thms.insert(index, thm)
        found_true_positive = True
        index -= 1
        continue
      assert response.HasField('goals'), 'response: %s' % response
      new_subgoals = list(response.goals.goals)
      no_match = False
      if len(new_subgoals) == len(tapp.subgoals):
        for i, sg in enumerate(new_subgoals):
          if not _matches_subgoal(sg, tapp.subgoals[i]):
            no_match = True
            break
      else:
        no_match = True
      if no_match:
        thms.insert(index, thm)
        found_true_positive = True
      else:
        if found_true_positive:
          false_positives.append(thm)
        else:
          other_negatives.append(thm)
        time_spent = elapsed_msecs
      index -= 1
    del tapp.parameters[0].theorems[:]
    tapp.parameters[0].theorems.extend(thms)
    tapp.parameters[0].hard_negative_theorems.extend(
        false_positives[:MAX_HARD_NEGATIVES])
    if len(false_positives) < MIN_HARD_NEGATIVES:
      other_negatives.reverse()
      tapp.parameters[0].hard_negative_theorems.extend(
          other_negatives[:(MIN_HARD_NEGATIVES - len(false_positives))])
    tapp.time_spent = time_spent

  def prune_tactic_applications(self, proof_node: deephol_pb2.ProofNode):
    for proof in proof_node.proofs:
      if proof.result == deephol_pb2.TacticApplication.SUCCESS:
        self.prune_tactic_application(proof_node.goal, proof)

  def prune_closed_tactic_applications(self, proof_node: deephol_pb2.ProofNode):
    for proof in proof_node.proofs:
      if proof.closed:
        assert proof.result == deephol_pb2.TacticApplication.SUCCESS
        self.prune_tactic_application(proof_node.goal, proof)
