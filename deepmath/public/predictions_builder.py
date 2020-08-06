# Lint as: python3
"""Builder for predictions objects."""
from deepmath.deephol import deephol_pb2
from deepmath.deephol import holparam_predictor
from deepmath.deephol import predictions


def build(prover_options: deephol_pb2.ProverOptions) -> predictions.Predictions:
  """Builds predictor for given prover options.

  Tries to load model architecture and graph representation from the checkpoint
  directory. If this fails, it defaults to using the fields from the prover
  options.

  Args:
    prover_options: prover options

  Returns:
    A predictor
  """
  return holparam_predictor.HolparamPredictor(
      ckpt=str(prover_options.path_model_prefix),
      max_embedding_batch_size=prover_options.max_embedding_batch_size,
      max_score_batch_size=prover_options.max_score_batch_size)
