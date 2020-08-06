"""Utility functions to generate mock proof logs for unit testing.

These functions are useful for unit testing ProofLog related code.
"""
from typing import List
from deepmath.deephol import deephol_pb2
from deepmath.deephol import theorem_utils
from deepmath.proof_assistant import proof_assistant_pb2


def make_goal(i: int) -> proof_assistant_pb2.Theorem:
  """Make a goal with content generated from the integer.

  Args:
    i: Index to uniquely identify the goal.

  Returns:
    The generated proof_assistant_pb2.Theorem object.
  """
  return proof_assistant_pb2.Theorem(
      assumptions=[theorem_utils.assume_term('h')],
      conclusion='c%d' % i,
      tag=proof_assistant_pb2.Theorem.GOAL)


def add_node(log: deephol_pb2.ProofLog,
             proofs: List[List[int]],
             proved: bool = True,
             root_goal: bool = False,
             prediction_time=37):
  """Add a new mock node to the proof log.

  Args:
    log: The proof log to be extended.
    proofs: Lists of lists of node indices representing the proof attempts
      (TacticApplications).
    proved: If true, the node is to be marked as true.
    root_goal: If true, the node is marked as a root goal, otherwise a subgoal.
    prediction_time: The time the action generator used to generate predictions.
  """
  node_index = len(log.nodes)
  status = proof_assistant_pb2.Theorem.UNKNOWN
  if proved:
    status = deephol_pb2.ProofNode.PROVED

  def make_tactic_application(proof):
    return deephol_pb2.TacticApplication(
        subgoals=[make_goal(i) for i in proof],
        closed=True,
        result=deephol_pb2.TacticApplication.SUCCESS)

  proofs = [make_tactic_application(proof) for proof in proofs]
  goal = make_goal(node_index)
  log.nodes.add(
      goal=goal,
      status=status,
      proofs=proofs,
      action_generation_time_millisec=prediction_time,
      root_goal=root_goal)


def new_log(num_proofs: int = 1, time_spent: int = 10000):
  """Create a new proof log."""
  return deephol_pb2.ProofLog(
      num_proofs=num_proofs,
      prover_options=deephol_pb2.ProverOptions(),
      time_spent=time_spent)
