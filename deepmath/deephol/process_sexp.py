"""A simple preprocessing helper."""

from typing import Optional, Text


def process_sexp(s: Text, truncate: Optional[int]):
  """Goal and theorem formatting function used for training."""
  # Remove the parentheses.
  s = s.replace('(', ' ').replace(')', ' ')
  if truncate is None:
    return ' '.join(s.split())
  # Truncate the string.
  return ' '.join(s.split()[:truncate])
