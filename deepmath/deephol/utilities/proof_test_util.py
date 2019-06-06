"""Utility functions to generate mock proof logs for unit testing.

These functions are useful for unit testing ProofLog related code.
"""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function
from typing import List
from deepmath.deephol import deephol_pb2
from deepmath.proof_assistant import proof_assistant_pb2


def make_theorem(i: int, theorem: bool = True) -> proof_assistant_pb2.Theorem:
  """Make a theorem or goal with content generated from the integer.

  Args:
    i: Index to uniquely identify the theorem.
    theorem: If true, then it will be tagged as a theorem, otherwise a goal.

  Returns:
    The generated proof_assistant_pb2.Theorem object.
  """
  if theorem:
    tag = proof_assistant_pb2.Theorem.THEOREM
  else:
    tag = proof_assistant_pb2.Theorem.GOAL
  return proof_assistant_pb2.Theorem(
      hypotheses=['h'], conclusion='c%d' % i, tag=tag)


def add_node(log: deephol_pb2.ProofLog,
             proofs: List[List[int]],
             proved: bool = True,
             theorem: bool = False,
             prediction_time=37):
  """Add a new mock node to the proof log.

  Args:
    log: The proof log to be extended.
    proofs: Lists of lists of node indices representing the proof attempts
      (TacticApplications).
    proved: If true, the node is to be marked as true.
    theorem: If true, the node is marked as a theorem, otherwise a subgoal.
    prediction_time: The time the action generator used to generate predictions.
  """
  node_index = len(log.nodes)
  status = proof_assistant_pb2.Theorem.UNKNOWN
  if proved:
    status = deephol_pb2.ProofNode.PROVED

  def make_tactic_application(proof):
    return deephol_pb2.TacticApplication(
        subgoals=[make_theorem(i) for i in proof],
        closed=True,
        result=deephol_pb2.TacticApplication.SUCCESS)

  proofs = [make_tactic_application(proof) for proof in proofs]
  log.nodes.add(
      goal=make_theorem(node_index, theorem=theorem),
      status=status,
      proofs=proofs,
      action_generation_time_millisec=prediction_time)


def new_log(num_proofs: int = 1, time_spent: int = 10000):
  """Create a new proof log."""
  return deephol_pb2.ProofLog(
      num_proofs=num_proofs,
      prover_options=deephol_pb2.ProverOptions(),
      time_spent=time_spent)
