"""HOL-light python API via gRPC.

This is the python API to communicate with a HOL-light server via gRPC.
"""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function
import grpc
from deepmath.proof_assistant import proof_assistant_pb2
from deepmath.proof_assistant import proof_assistant_pb2_grpc


class HolLight(object):
  """Class for intefacing a HOL Light prover."""

  def __init__(self):
    self.channel = grpc.insecure_channel('localhost:50051')
    self.stub = proof_assistant_pb2_grpc.ProofAssistantServiceStub(self.channel)

  def ApplyTacticToGoal(self, request: proof_assistant_pb2.ApplyTacticRequest
                       ) -> proof_assistant_pb2.ApplyTacticResponse:
    return self.stub.ApplyTactic(request)

  def VerifyProof(self, request: proof_assistant_pb2.VerifyProofRequest
                 ) -> proof_assistant_pb2.VerifyProofResponse:
    return self.stub.VerifyProof(request)

  def RegisterTheorem(self, request: proof_assistant_pb2.RegisterTheoremRequest
                     ) -> proof_assistant_pb2.RegisterTheoremResponse:
    return self.stub.RegisterTheorem(request)
