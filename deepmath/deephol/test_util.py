"""Utilities for testing."""
import os
from absl import flags

FLAGS = flags.FLAGS


def test_src_dir_path(relative_path):
  """Creates an absolute test srcdir path given a relative path.

  Args:
    relative_path: a path relative to deepmath root. e.g. "deephol/test_data/".

  Returns:
    An absolute path to the linked in runfiles.
  """
  return os.path.join(os.environ['TEST_SRCDIR'],
                      'deepmath/deepmath/', relative_path)
