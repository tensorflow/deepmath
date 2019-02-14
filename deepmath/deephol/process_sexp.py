"""A simple preprocessing helper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

TRUNCATE = 1000


def process_sexp(s):
  """Goal and theorem formatting function used for training."""
  # Remove the parens.
  s = s.replace('(', ' ').replace(')', ' ')
  # Truncate the string.
  return ' '.join(s.split()[:TRUNCATE])
