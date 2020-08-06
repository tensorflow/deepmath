"""Module for functions that convert text-based terms into theorems."""

from typing import Optional, Text
import tensorflow.compat.v1 as tf
from deepmath.deephol import theorem_fingerprint
from deepmath.proof_assistant import proof_assistant_pb2


def assume_term(term: Text, assumption_index: Optional[int] = None):
  """Convert hypothesis term A into theorem format: {A} |- A."""
  res = proof_assistant_pb2.Theorem(
      hypotheses=[term],
      conclusion=term,
      tag=proof_assistant_pb2.Theorem.THEOREM)
  if assumption_index is not None:
    res.assumption_index = assumption_index
  return res


def theorem_to_goal(
    theorem: proof_assistant_pb2.Theorem) -> proof_assistant_pb2.Theorem:
  """Creates a goal from an input theorem.

  Args:
    theorem: Input theorem.

  Returns:
    Theorem proto of type GOAL otherwise equivalent to the input theorem.
  """
  goal = proof_assistant_pb2.Theorem()
  goal.CopyFrom(theorem)  # deep-copy
  goal.tag = proof_assistant_pb2.Theorem.GOAL
  convert_legacy_goal(goal)
  if goal.HasField('definition'):
    tf.logging.warning('Converted a definition to a goal.')
  if goal.HasField('type_definition'):
    tf.logging.warning('Converted a type definition to a goal.')
  return goal


def convert_legacy_goal(goal: proof_assistant_pb2.Theorem):
  """We represented assumptions of goals as hypotheses.

  This method converts these hypotheses A in goals to assumptions A |- A.

  Args:
    goal: The goal that potentially has legacy hypotheses.
  """
  if goal.tag != proof_assistant_pb2.Theorem.GOAL:
    # TODO(b/139834654): Make this a fatal error.
    tf.logging.error('Fixing goal in proof log with incorrect tag.')
    goal.tag = proof_assistant_pb2.Theorem.GOAL
  if goal.hypotheses:
    # convert "A|-B" to "[A|-A], B".
    for hyp in goal.hypotheses:
      goal.assumptions.extend([assume_term(hyp)])
    goal.ClearField('hypotheses')
    goal.ClearField('fingerprint')
  if not goal.HasField('fingerprint') and goal.HasField('conclusion'):
    goal.fingerprint = theorem_fingerprint.Fingerprint(goal)
  for index, assum in enumerate(goal.assumptions):
    convert_legacy_theorem(assum)
    if assum.HasField('assumption_index') and assum.assumption_index != index:
      tf.logging.error('Assumption index was set incorrectly.')
    assum.assumption_index = index


def convert_legacy_theorem(theorem: proof_assistant_pb2.Theorem):
  """Handles legacy theorems."""
  theorem.tag = proof_assistant_pb2.Theorem.THEOREM
  if not theorem.HasField('fingerprint'):
    theorem.fingerprint = theorem_fingerprint.Fingerprint(theorem)
