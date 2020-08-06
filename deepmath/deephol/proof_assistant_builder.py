"""Builder for proof assistant objects."""
import tensorflow.compat.v1 as tf
from deepmath.public import proof_assistant
from deepmath.proof_assistant import proof_assistant_pb2


def build(
    theorem_database: proof_assistant_pb2.TheoremDatabase
) -> proof_assistant.ProofAssistant:
  """Starts up HOL and seeds it with given TheoremDatabase."""
  tf.logging.info('Setting up and registering theorems with proof assistant...')
  proof_assistant_obj = proof_assistant.ProofAssistant()
  for thm in theorem_database.theorems:
    response = proof_assistant_obj.RegisterTheorem(
        proof_assistant_pb2.RegisterTheoremRequest(theorem=thm))
    if response.HasField('error_msg') and response.error_msg:
      raise ValueError('Registration failed for %d with: %s' %
                       (response.fingerprint, response.error_msg))
  tf.logging.info('Proof assistant setup done.')
  return proof_assistant_obj
