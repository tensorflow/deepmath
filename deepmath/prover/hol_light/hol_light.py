"""HOL-light python API.

This is the python API (without implementation) to communicate with the HOL
Light sandbox or docker image.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function
from typing import Text
from deepmath.prover.hol_light import error
from deepmath.prover.hol_light import hol_light_pb2

NOT_IMPLEMENTED_STATUS = error.StatusNotOK('Not implemented')


class HolLight(object):
  """Class for intefacing a HOL Light prover."""

  class Options(object):
    """Options for initializing the HOL Light API."""

    def __init__(self):
      self.show_stdout = False
      self.check_builtin_theorems = False
      self.theorem_database = None

    def ShowStdout(self) -> Options:
      self.show_stdout = True
      return self

    def CheatBuiltinTheorems(self) -> Options:
      self.cheat_builtin_theorems = True
      return self

    def SetTheoremDatabase(
        self, theorem_database: hol_light_pb2.TheoremDatabase) -> Options:
      self.theorem_database = theorem_database
      return self

  def __init__(self, options):
    self.options = options

  def ApplyTacticToGoal(self, request: hol_light_pb2.ApplyTacticRequest
                       ) -> hol_light_pb2.ApplyTacticResponse:
    del request
    raise NOT_IMPLEMENTED_STATUS

  def SetGoal(self, term: Goal):
    del term
    raise NOT_IMPLEMENTED_STATUS

  def GetGoals(self) -> hol_light_pb2.GoalList:
    raise NOT_IMPLEMENTED_STATUS

  def RotateGoals(self, n: int):
    del n
    raise NOT_IMPLEMENTED_STATUS

  def ApplyTactic(self, tactic: Text):
    del tactic
    raise NOT_IMPLEMENTED_STATUS

  def Undo(self):
    raise NOT_IMPLEMENTED_STATUS

  def RegisterLastTheorem(self):
    raise NOT_IMPLEMENTED_STATUS

  def CheatTheorem(self, theorem: Theorem):
    del theorem
    raise NOT_IMPLEMENTED_STATUS

  def Define(self, definition: Definition):
    del definition
    raise NOT_IMPLEMENTED_STATUS

  def DefineType(self, definition: hol_light_pb2.TypeDefinition):
    del definition
    raise NOT_IMPLEMENTED_STATUS

  def SetTermEncoding(self, enc: hol_light_pb2.TermEncoding):
    del enc
    raise NOT_IMPLEMENTED_STATUS

  def VerifyProof(self, request: hol_light_pb2.VerifyProofRequest
                 ) -> hol_light_pb2.VerifyProofResponse:
    del request
    raise NOT_IMPLEMENTED_STATUS

  @staticmethod
  def Create(options: Options) -> HolLight:
    """Compatibility method to match the internal C++ API."""
    return HolLight(options)
