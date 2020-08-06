# Lint as: python3
"""Tests for deepmath.deephol.utilities.sexpression_de_bruijn_indexer."""
from absl.testing import parameterized
import tensorflow.compat.v1 as tf
from deepmath.deephol.utilities import sexpression_de_bruijn_indexer


class SexpressionDeBruijnIndexerTest(parameterized.TestCase):

  @parameterized.parameters(
      # No bound variables.
      ('(c (fun (bool) (fun (bool) (bool))) /\\)',
       '(c (fun (bool) (fun (bool) (bool))) /\\)'),
      # Single bound variable.
      ('(l (v T x) (v T x))', '(l (v T 1) (v T 1))'),
      # Free variable with the same name as a bound variable.
      ('(l (v T x) (v TT x))', '(l (v T 1) (v TT x))'),
      # Nested variable bindings with the same name and type.
      ('(l (v T x) (l (v T x) (v T x)))', '(l (v T 2) (l (v T 1) (v T 1)))'),
      # Different variable bindings at the same nesting level.
      ('(a (a (c (fun (fun T T) (fun (fun T T) (bool))) f) (l (v T x) (v T x))) (l (v T y) (v T y)))',
       '(a (a (c (fun (fun T T) (fun (fun T T) (bool))) f) (l (v T 1) (v T 1))) (l (v T 1) (v T 1)))'
      ),
      # Different nesting levels in a combination.
      ('(a (a (c (fun (fun T T) (fun (fun T (fun T T)) (bool))) f) (l (v T x) (v T x))) (l (v T x) (l (v T x) (v T x))))',
       '(a (a (c (fun (fun T T) (fun (fun T (fun T T)) (bool))) f) (l (v T 1) (v T 1))) (l (v T 2) (l (v T 1) (v T 1))))'
      ),
      # Different nesting levels in an abstraction.
      ('(l (v T x) (a (a (c (fun (fun T T) (fun (fun T (fun T T)) (bool))) f) (l (v T x) (v T x))) (l (v T x) (l (v T x) (v T x)))))',
       '(l (v T 3) (a (a (c (fun (fun T T) (fun (fun T (fun T T)) (bool))) f) (l (v T 1) (v T 1))) (l (v T 2) (l (v T 1) (v T 1)))))'
      ),
      # Variables bound outside of the current abstraction.
      ('(l (v T y) (a (a (c (fun (fun T T) (fun (fun T (fun T T)) (bool))) f) (l (v T x) (v T x))) (l (v T x) (l (v T x) (v T y)))))',
       '(l (v T 3) (a (a (c (fun (fun T T) (fun (fun T (fun T T)) (bool))) f) (l (v T 1) (v T 1))) (l (v T 2) (l (v T 1) (v T 3)))))'
      ),
      # Empty S-expression.
      ('', ''),
  )
  def test_renaming(self, sexp, expected_result):
    result = sexpression_de_bruijn_indexer.rename_bound_variables(sexp)
    self.assertEqual(expected_result, result)

if __name__ == '__main__':
  tf.test.main()
