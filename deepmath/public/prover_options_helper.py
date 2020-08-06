"""Update prover options based on the training options.

Utility to figure out the model architecture based on the training options.
"""
from typing import Text
from deepmath.deephol import deephol_pb2


def load_training_options(unused_prover_options):
  """Assumes model_architecture is already set."""
  pass


def get_default_prover_options(
    unused_model_path: Text) -> deephol_pb2.ProverOptions:
  """Gets default prover options from file."""
  raise NotImplementedError()
