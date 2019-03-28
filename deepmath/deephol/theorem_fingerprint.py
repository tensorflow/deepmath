"""Theorem fingerprint reimplementation in python.

This file contains a backward compatible version of theorem_fingerprint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import farmhash

# We cut off the two last significant bits for ocaml compatibility.
MASK62 = ((1 << 62) - 1)
MASK64 = ((1 << 64) - 1)
MUL = 0x9ddfea08eb382d69


def _PairFingerprint(low, high):
  """This is a reimplementation of the 128 -> 64 fingerprint of farmhash.

  Args:
    low: 64 bit unsigned integer value
    high: 64 bit unsigned integer value

  Returns:
    64 bit unsigned integer value
  """
  a = ((low ^ high) * MUL) & MASK64
  a ^= (a >> 47)
  b = ((high ^ a) * MUL) & MASK64
  b ^= (b >> 44)
  b = (b * MUL) & MASK64
  b ^= (b >> 41)
  return (b * MUL) & MASK64


def Fingerprint(theorem):
  """Compute a unique, stable fingerprint for theorem objects.

  Args:
    theorem: proof_assistant_pb2.Theorem object

  Returns:
    62 bit non-negative integer fingerprint. Note that we truncate to 62 bits
    for OCaml compatibility. OCaml uses 63 bit signed integers.
  """
  if not theorem.HasField('conclusion') and theorem.HasField('fingerprint'):
    return theorem.fingerprint
  fp = farmhash.fingerprint64(theorem.conclusion)
  for hypothesis in theorem.hypotheses:
    tmp = farmhash.fingerprint64(hypothesis)
    fp = _PairFingerprint(fp, tmp)
  result = fp & MASK62
  assert (not theorem.HasField('fingerprint') or
          theorem.fingerprint == result), (
              'Inconsistent fingerprints %d != %d in Theorem protobuf.' %
              (result, theorem.fingerprint))
  return result


def ToTacticArgument(theorem):
  """Return a representation of the theorem as a tactic argument label.

  Args:
    theorem: proof_assistant_pb2.Theorem object

  Returns:
    String that can be used as a tactic argument.
  """
  return 'THM %d' % Fingerprint(theorem)
