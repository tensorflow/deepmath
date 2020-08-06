"""Seq2Seq based action generator.

Uses a transformer based translation mode to predict actions.
"""
from typing import List, Text
from deepmath.deephol import abstract_action_generator
from deepmath.deephol import deephol_pb2
from deepmath.deephol import proof_search_tree
from deepmath.proof_assistant import proof_assistant_pb2

Suggestion = abstract_action_generator.Suggestion


class SequenceActionGenerator(abstract_action_generator.AbstractActionGenerator
                             ):
  """Not implemented in the open source version."""

  def __init__(self, theorem_database: proof_assistant_pb2.TheoremDatabase,
               tactics: List[deephol_pb2.Tactic],
               options: deephol_pb2.SequenceActionGeneratorOptions,
               model_dir: Text):
    raise NotImplementedError()

  def _step(self, node: proof_search_tree.ProofSearchNode,
            thm_number: int) -> List[Suggestion]:
    """Generates a list of possible ApplyTactic argument strings from a goal.

    Args:
      node: state of the proof search, starting at current goal.
      thm_number: Index before which we can use any theorem in the theorem
        database.

    Raises:
      NotImplementedError.
    """
    raise NotImplementedError()
