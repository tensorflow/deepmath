"""A python client interface for ProofAssistantService."""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function
import grpc
import tensorflow as tf
from deepmath.proof_assistant import proof_assistant_pb2
from deepmath.proof_assistant import proof_assistant_pb2_grpc

tf.flags.DEFINE_string(
    'proof_assistant_server_address', 'localhost:2000',
    'address (including port) of the proof assistant server')

FLAGS = tf.flags.FLAGS

GIGABYTE = 1024 * 1024 * 1024
GRPC_MAX_MESSAGE_LENGTH = GIGABYTE


class ProofAssistant(object):
  """Class for intefacing a proof assistant."""

  def __init__(self):
    self.channel = grpc.insecure_channel(
        FLAGS.proof_assistant_server_address,
        options=[('grpc.max_send_message_length', GRPC_MAX_MESSAGE_LENGTH),
                 ('grpc.max_receive_message_length', GRPC_MAX_MESSAGE_LENGTH)])
    self.stub = proof_assistant_pb2_grpc.ProofAssistantServiceStub(self.channel)

  def ApplyTactic(self, request: proof_assistant_pb2.ApplyTacticRequest
                 ) -> proof_assistant_pb2.ApplyTacticResponse:
    return self.stub.ApplyTactic(request)

  def VerifyProof(self, request: proof_assistant_pb2.VerifyProofRequest
                 ) -> proof_assistant_pb2.VerifyProofResponse:
    return self.stub.VerifyProof(request)

  def RegisterTheorem(self, request: proof_assistant_pb2.RegisterTheoremRequest
                     ) -> proof_assistant_pb2.RegisterTheoremResponse:
    return self.stub.RegisterTheorem(request)
