"""Theorem fingerprint reimplementation in python.

This file contains a backward compatible version of theorem_fingerprint.
"""
import itertools
from typing import Text
import farmhash
from deepmath.proof_assistant import proof_assistant_pb2

# We cut off the two last significant bits for ocaml compatibility.
MASK62 = ((1 << 62) - 1)
MASK64 = ((1 << 64) - 1)
MUL = 0x9ddfea08eb382d69


def _PairFingerprint(low: int, high: int) -> int:
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


def Fingerprint(theorem: proof_assistant_pb2.Theorem) -> int:
  """Compute a unique, stable fingerprint for theorem objects.

  Args:
    theorem: proof_assistant_pb2.Theorem object

  Returns:
    62 bit non-negative integer fingerprint. Note that we truncate to 62 bits
    for OCaml compatibility. OCaml uses 63 bit signed integers.
  """
  if theorem.hypotheses and theorem.assumptions:
    raise ValueError(
        'Theorem proto cannot have both hypotheses and assumptions.')
  if theorem.assumptions and theorem.tag != proof_assistant_pb2.Theorem.GOAL:
    raise ValueError('Only theorem protos with tag GOAL can have assumptions.')
  if not theorem.HasField('conclusion') and theorem.HasField('fingerprint'):
    return theorem.fingerprint
  fp = farmhash.fingerprint64(theorem.conclusion)
  for hypothesis in theorem.hypotheses:
    tmp = farmhash.fingerprint64(hypothesis)
    fp = _PairFingerprint(fp, tmp)
  for assumption in theorem.assumptions:
    if assumption.tag == proof_assistant_pb2.Theorem.GOAL:
      raise ValueError('Assumption of goal must be actual theorem; not GOAL.')
    tmp = Fingerprint(assumption)
    tmp += 1  # Ensures that "[t1 |- t2], t3", "[|-t1, |-t2], t3" are different
    fp = _PairFingerprint(fp, tmp)
  result = fp & MASK62
  assert (not theorem.HasField('fingerprint') or
          theorem.fingerprint == result), (
              'Inconsistent fingerprints %d != %d in Theorem protobuf.' %
              (result, theorem.fingerprint))
  return result


def ToTacticArgument(theorem: proof_assistant_pb2.Theorem) -> Text:
  """Return a representation of the theorem as a tactic argument label.

  Args:
    theorem: proof_assistant_pb2.Theorem object

  Returns:
    String that can be used as a tactic argument.
  """
  return 'THM %d' % Fingerprint(theorem)


SEPARATOR_FINGERPRINT = 54321  # separates goals and targets and premise sets


def PremiseSetFingerprint(premise_set: proof_assistant_pb2.PremiseSet) -> int:
  """Computes the fingerprint of a PremiseSet."""
  if premise_set.reference_sets:
    raise NotImplementedError
  result = farmhash.fingerprint64('empty premise set')
  for section in premise_set.sections:
    if not (section.HasField('database_name') and
            section.HasField('before_premise')):
      raise ValueError('Db section needs database_name and before_premise.')
    result = _PairFingerprint(result,
                              farmhash.fingerprint64(section.database_name))
    result = _PairFingerprint(result, section.before_premise)
  return result & MASK62


def TaskFingerprint(prover_task: proof_assistant_pb2.ProverTask) -> int:
  """Fingerprint of the prover task; different from theorem fingerprints."""
  goal_fingerprints = map(Fingerprint, prover_task.goals)
  target_fingerprints = map(Fingerprint, prover_task.targets)
  premise_set_fingerprints = []
  if prover_task.HasField('premise_set'):
    premise_set_fingerprints.append(
        PremiseSetFingerprint(prover_task.premise_set))
  combined_fingerprint = None
  for fp in itertools.chain(goal_fingerprints, [SEPARATOR_FINGERPRINT],
                            target_fingerprints, [SEPARATOR_FINGERPRINT],
                            premise_set_fingerprints):
    if combined_fingerprint is None:
      combined_fingerprint = fp
    else:
      combined_fingerprint = _PairFingerprint(combined_fingerprint, fp)
  return combined_fingerprint & MASK62  # pytype: disable=bad-return-type
