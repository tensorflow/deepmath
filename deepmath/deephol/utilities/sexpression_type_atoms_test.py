# Lint as: python3
"""Tests for deepmath.deephol.utilities.sexpression_type_atoms."""

from absl.testing import parameterized
import tensorflow.compat.v1 as tf
from deepmath.deephol.utilities import hol_light_sexpression_syntax
from deepmath.deephol.utilities import sexpression_type_atoms


class SexpressionTypeAtomsTest(parameterized.TestCase):

  @parameterized.parameters(
      ('(c (fun (bool) (fun (bool) (bool))) /\\)',
       '(c (<TYPE> fun (<TYPE> bool) (<TYPE> fun (<TYPE> bool) (<TYPE> bool))) /\\)'
      ),
      ('(v (fun (bool) (fun (bool) (bool))) x)',
       '(v (<TYPE> fun (<TYPE> bool) (<TYPE> fun (<TYPE> bool) (<TYPE> bool))) x)'
      ),
      ('(l (v (bool) x) (v (bool) x))',
       '(l (v (<TYPE> bool) x) (v (<TYPE> bool) x))'),
      ('(a (c (fun (fun (bool) (bool)) (bool)) f) (l (v (bool) x) (v (bool) x)))',
       '(a (c (<TYPE> fun (<TYPE> fun (<TYPE> bool) (<TYPE> bool)) (<TYPE> bool)) f) (l (v (<TYPE> bool) x) (v (<TYPE> bool) x)))'
      ),
      ('(a (c (fun (fun T T) (bool)) f) (l (v T x) (v T x)))',
       '(a (c (<TYPE> fun (<TYPE> fun T T) (<TYPE> bool)) f) (l (v T x) (v T x)))'
      ),
      # Empty S-expression.
      ('', ''),
  )
  def test_type_atoms_for_terms(self, sexp, expected_result):
    result = sexpression_type_atoms.insert_type_atoms(
        sexp, hol_light_sexpression_syntax.SyntaxElementKind.UNKNOWN)
    self.assertEqual(expected_result, result)


if __name__ == '__main__':
  tf.test.main()
