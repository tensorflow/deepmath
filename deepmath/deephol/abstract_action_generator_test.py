"""Tests for deepmath.deephol.abstract_action_generator."""

from typing import List
import tensorflow.compat.v1 as tf

from deepmath.deephol import abstract_action_generator
from deepmath.deephol import deephol_pb2
from deepmath.deephol import proof_search_tree
from deepmath.proof_assistant import proof_assistant_pb2


class DummyGenerator(abstract_action_generator.AbstractActionGenerator):

  def __init__(self, theorem_database: proof_assistant_pb2.TheoremDatabase,
               tactics: List[deephol_pb2.Tactic]):
    pass
    # super(DummyGenerator, self).__init__(theorem_database, tactics)

  def _step(self, node: proof_search_tree.ProofSearchNode,
            thm_number: int) -> List[abstract_action_generator.Suggestion]:
    return []


class AbstractActionGeneratorTest(tf.test.TestCase):

  def setUp(self):
    tf.test.TestCase.setUp(self)
    pass

  def testCanBuild(self):
    _ = DummyGenerator(None, [])


if __name__ == "__main__":
  tf.test.main()
