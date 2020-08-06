# Lint as: python3
"""Tests for deepmath.deephol.utilities.hol_light_sexpression_graphs."""

from absl.testing import parameterized
import tensorflow.compat.v1 as tf
from deepmath.deephol.utilities import hol_light_sexpression_graphs
from deepmath.deephol.utilities import hol_light_sexpression_syntax as hol
from deepmath.deephol.utilities import sexpression_graphs


class SexpressionTypeAtomsTest(parameterized.TestCase):

  @parameterized.parameters(
      ('A', hol.SyntaxElementKind.TYPE, True),
      ('(bool)', hol.SyntaxElementKind.TYPE, False),
      ('(fun X Y)', hol.SyntaxElementKind.TYPE, True),
      ('(fun (bool) (bool))', hol.SyntaxElementKind.TYPE, False),
      ('(fun (bool) (fun X (bool)))', hol.SyntaxElementKind.TYPE, True),
      ('(fun (bool) (fun (bool) (bool)))', hol.SyntaxElementKind.TYPE, False),
      ('(c (bool) T)', hol.SyntaxElementKind.TERM, False),
      ('(c A x)', hol.SyntaxElementKind.TERM, True),
      ('(c (fun (bool) (bool)) ~)', hol.SyntaxElementKind.TERM, False),
      ('(c (fun A (bool)) x)', hol.SyntaxElementKind.TERM, True),
      ('(v (bool) x)', hol.SyntaxElementKind.TERM, False),
      ('(v A x)', hol.SyntaxElementKind.TERM, True),
      ('(v (fun (bool) (bool)) x)', hol.SyntaxElementKind.TERM, False),
      ('(v (fun A (bool)) x)', hol.SyntaxElementKind.TERM, True),
      ('(l (v (bool) x) (c (bool) T))', hol.SyntaxElementKind.TERM, False),
      ('(l (v A x) (c (bool) T))', hol.SyntaxElementKind.TERM, True),
      ('(l (v (bool) x) (c A y))', hol.SyntaxElementKind.TERM, True),
      ('(a (c (fun (bool) (bool)) ~) (c (bool) T))', hol.SyntaxElementKind.TERM,
       False),
      ('(a (c (fun (bool) A) x) (c (bool) T))', hol.SyntaxElementKind.TERM,
       True),
      ('(a (c (fun (bool) (bool)) ~) (a (c (fun A (bool)) x) (c A y)))',
       hol.SyntaxElementKind.TERM, True),
      ('(g () (c (bool) T))', hol.SyntaxElementKind.UNKNOWN, False),
      ('(g () (a (c (fun A (bool)) x) (c A y)))', hol.SyntaxElementKind.UNKNOWN,
       True),
      ('a', hol.SyntaxElementKind.VARIABLE_NAME, False),
      ('a', hol.SyntaxElementKind.CONSTANT_NAME, False),
      ('a', hol.SyntaxElementKind.TYPE_CONSTRUCTOR_NAME, False),
      ('a', hol.SyntaxElementKind.SEXPRESSION_KIND, False),
  )
  def test_has_type_variables(self, sexp, node_kind, expected_result):
    sexp_graph = sexpression_graphs.SExpressionGraph(sexp)
    hol_light_graph = hol_light_sexpression_graphs.HolLightSExpressionGraph(
        sexp_graph, has_type_atoms=False)
    node = sexpression_graphs.to_node_id(sexp)
    result = hol_light_graph.has_type_variables(node, node_kind)
    self.assertEqual(expected_result, result)

  @parameterized.parameters(
      ('A', hol.SyntaxElementKind.TYPE, True),
      ('(<TYPE> bool)', hol.SyntaxElementKind.TYPE, False),
      ('(<TYPE> fun X Y)', hol.SyntaxElementKind.TYPE, True),
      ('(<TYPE> fun (<TYPE> bool) (<TYPE> bool))', hol.SyntaxElementKind.TYPE,
       False),
      ('(<TYPE> fun (<TYPE> bool) (<TYPE> fun X (<TYPE> bool)))',
       hol.SyntaxElementKind.TYPE, True),
      ('(<TYPE> fun (<TYPE> bool) (<TYPE> fun (<TYPE> bool) (<TYPE> bool)))',
       hol.SyntaxElementKind.TYPE, False),
      ('(c (<TYPE> bool) T)', hol.SyntaxElementKind.TERM, False),
      ('(c A x)', hol.SyntaxElementKind.TERM, True),
      ('(c (<TYPE> fun (<TYPE> bool) (<TYPE> bool)) ~)',
       hol.SyntaxElementKind.TERM, False),
      ('(c (<TYPE> fun A (<TYPE> bool)) x)', hol.SyntaxElementKind.TERM, True),
      ('(v (<TYPE> bool) x)', hol.SyntaxElementKind.TERM, False),
      ('(v A x)', hol.SyntaxElementKind.TERM, True),
      ('(v (<TYPE> fun (<TYPE> bool) (<TYPE> bool)) x)',
       hol.SyntaxElementKind.TERM, False),
      ('(v (<TYPE> fun A (<TYPE> bool)) x)', hol.SyntaxElementKind.TERM, True),
      ('(l (v (<TYPE> bool) x) (c (<TYPE> bool) T))',
       hol.SyntaxElementKind.TERM, False),
      ('(l (v A x) (c (<TYPE> bool) T))', hol.SyntaxElementKind.TERM, True),
      ('(l (v (<TYPE> bool) x) (c A y))', hol.SyntaxElementKind.TERM, True),
      ('(a (c (<TYPE> fun (<TYPE> bool) (<TYPE> bool)) ~) (c (<TYPE> bool) T))',
       hol.SyntaxElementKind.TERM, False),
      ('(a (c (<TYPE> fun (<TYPE> bool) A) x) (c (<TYPE> bool) T))',
       hol.SyntaxElementKind.TERM, True),
      ('(a (c (<TYPE> fun (<TYPE> bool) (<TYPE> bool)) ~) (a (c (<TYPE> fun A (<TYPE> bool)) x) (c A y)))',
       hol.SyntaxElementKind.TERM, True),
      ('(g () (c (<TYPE> bool) T))', hol.SyntaxElementKind.UNKNOWN, False),
      ('(g () (a (c (<TYPE> fun A (<TYPE> bool)) x) (c A y)))',
       hol.SyntaxElementKind.UNKNOWN, True),
      ('a', hol.SyntaxElementKind.VARIABLE_NAME, False),
      ('a', hol.SyntaxElementKind.CONSTANT_NAME, False),
      ('a', hol.SyntaxElementKind.TYPE_CONSTRUCTOR_NAME, False),
      ('a', hol.SyntaxElementKind.SEXPRESSION_KIND, False),
      ('<TYPE>', hol.SyntaxElementKind.TYPE_ATOM, False),
  )
  def test_has_type_variables_with_type_atoms(self, sexp, node_kind,
                                              expected_result):
    sexp_graph = sexpression_graphs.SExpressionGraph(sexp)
    hol_light_graph = hol_light_sexpression_graphs.HolLightSExpressionGraph(
        sexp_graph, has_type_atoms=True)
    node = sexpression_graphs.to_node_id(sexp)
    result = hol_light_graph.has_type_variables(node, node_kind)
    self.assertEqual(expected_result, result)

  @parameterized.parameters(
      ('A', hol.SyntaxElementKind.TYPE, []),
      ('(v (bool) x)', hol.SyntaxElementKind.TERM, ['(v (bool) x)']),
      ('(l (v (bool) x) (v (bool) x))', hol.SyntaxElementKind.TERM, []),
      ('(l (v (bool) x) (v (bool) y))', hol.SyntaxElementKind.TERM,
       ['(v (bool) y)']),
      ('(c (bool) x)', hol.SyntaxElementKind.TERM, []),
      ('(g () (v (bool) x))', hol.SyntaxElementKind.UNKNOWN, ['(v (bool) x)']),
      ('(v (bool) x)', hol.SyntaxElementKind.TYPE, []),
      ('a', hol.SyntaxElementKind.VARIABLE_NAME, []),
      ('a', hol.SyntaxElementKind.CONSTANT_NAME, []),
      ('a', hol.SyntaxElementKind.TYPE_CONSTRUCTOR_NAME, []),
      ('a', hol.SyntaxElementKind.SEXPRESSION_KIND, []),
  )
  def test_get_free_variables(self, sexp, node_kind, expected_result):
    sexp_graph = sexpression_graphs.SExpressionGraph(sexp)
    hol_light_graph = hol_light_sexpression_graphs.HolLightSExpressionGraph(
        sexp_graph, has_type_atoms=False)
    node = sexpression_graphs.to_node_id(sexp)
    result = hol_light_graph.get_free_variables(node, node_kind)
    self.assertEqual(
        frozenset(expected_result),
        frozenset(
            [sexp_graph.to_text(variable_node) for variable_node in result]))

  @parameterized.parameters(
      ('A', hol.SyntaxElementKind.TYPE, []),
      ('(v (<TYPE> bool) x)', hol.SyntaxElementKind.TERM,
       ['(v (<TYPE> bool) x)']),
      ('(l (v (<TYPE> bool) x) (v (<TYPE> bool) x))',
       hol.SyntaxElementKind.TERM, []),
      ('(l (v (<TYPE> bool) x) (v (<TYPE> bool) y))',
       hol.SyntaxElementKind.TERM, ['(v (<TYPE> bool) y)']),
      ('(c (<TYPE> bool) x)', hol.SyntaxElementKind.TERM, []),
      ('(g () (v (<TYPE> bool) x))', hol.SyntaxElementKind.UNKNOWN,
       ['(v (<TYPE> bool) x)']),
      ('(v (<TYPE> bool) x)', hol.SyntaxElementKind.TYPE, []),
      ('a', hol.SyntaxElementKind.VARIABLE_NAME, []),
      ('a', hol.SyntaxElementKind.CONSTANT_NAME, []),
      ('a', hol.SyntaxElementKind.TYPE_CONSTRUCTOR_NAME, []),
      ('a', hol.SyntaxElementKind.SEXPRESSION_KIND, []),
      ('<TYPE>', hol.SyntaxElementKind.TYPE_ATOM, []),
  )
  def test_get_free_variables_with_type_atoms(self, sexp, node_kind,
                                              expected_result):
    sexp_graph = sexpression_graphs.SExpressionGraph(sexp)
    hol_light_graph = hol_light_sexpression_graphs.HolLightSExpressionGraph(
        sexp_graph, has_type_atoms=True)
    node = sexpression_graphs.to_node_id(sexp)
    result = hol_light_graph.get_free_variables(node, node_kind)
    self.assertEqual(
        frozenset(expected_result),
        frozenset(
            [sexp_graph.to_text(variable_node) for variable_node in result]))

  @parameterized.parameters(
      ('A', hol.SyntaxElementKind.TYPE, []),
      ('(bool)', hol.SyntaxElementKind.TYPE, [
          ('bool', hol.SyntaxElementKind.TYPE_CONSTRUCTOR_NAME)
      ]),
      ('(fun x y)', hol.SyntaxElementKind.TYPE, [
          ('fun', hol.SyntaxElementKind.TYPE_CONSTRUCTOR_NAME),
          ('x', hol.SyntaxElementKind.TYPE), ('y', hol.SyntaxElementKind.TYPE)
      ]),
      ('(c (bool) T)', hol.SyntaxElementKind.TERM, [
          ('c', hol.SyntaxElementKind.SEXPRESSION_KIND),
          ('(bool)', hol.SyntaxElementKind.TYPE),
          ('T', hol.SyntaxElementKind.CONSTANT_NAME)
      ]),
      ('(v (bool) x)', hol.SyntaxElementKind.TERM, [
          ('v', hol.SyntaxElementKind.SEXPRESSION_KIND),
          ('(bool)', hol.SyntaxElementKind.TYPE),
          ('x', hol.SyntaxElementKind.VARIABLE_NAME)
      ]),
      ('(l (v (bool) x) (v (bool) x))', hol.SyntaxElementKind.TERM, [
          ('l', hol.SyntaxElementKind.SEXPRESSION_KIND),
          ('(v (bool) x)', hol.SyntaxElementKind.TERM),
          ('(v (bool) x)', hol.SyntaxElementKind.TERM)
      ]),
      ('(a (v (fun (bool) (bool)) x) (v (bool) y))', hol.SyntaxElementKind.TERM,
       [('a', hol.SyntaxElementKind.SEXPRESSION_KIND),
        ('(v (fun (bool) (bool)) x)', hol.SyntaxElementKind.TERM),
        ('(v (bool) y)', hol.SyntaxElementKind.TERM)]),
      ('(c (bool) T)', hol.SyntaxElementKind.UNKNOWN, [
          ('c', hol.SyntaxElementKind.SEXPRESSION_KIND),
          ('(bool)', hol.SyntaxElementKind.TYPE),
          ('T', hol.SyntaxElementKind.CONSTANT_NAME)
      ]),
      ('(v (bool) x)', hol.SyntaxElementKind.UNKNOWN, [
          ('v', hol.SyntaxElementKind.SEXPRESSION_KIND),
          ('(bool)', hol.SyntaxElementKind.TYPE),
          ('x', hol.SyntaxElementKind.VARIABLE_NAME)
      ]),
      ('(l (v (bool) x) (v (bool) x))', hol.SyntaxElementKind.UNKNOWN, [
          ('l', hol.SyntaxElementKind.SEXPRESSION_KIND),
          ('(v (bool) x)', hol.SyntaxElementKind.TERM),
          ('(v (bool) x)', hol.SyntaxElementKind.TERM)
      ]),
      ('(a (v (fun (bool) (bool)) x) (v (bool) y))',
       hol.SyntaxElementKind.UNKNOWN, [
           ('a', hol.SyntaxElementKind.SEXPRESSION_KIND),
           ('(v (fun (bool) (bool)) x)', hol.SyntaxElementKind.TERM),
           ('(v (bool) y)', hol.SyntaxElementKind.TERM)
       ]),
      ('(g () (v (bool) x))', hol.SyntaxElementKind.UNKNOWN, [
          ('g', hol.SyntaxElementKind.UNKNOWN),
          ('()', hol.SyntaxElementKind.UNKNOWN),
          ('(v (bool) x)', hol.SyntaxElementKind.UNKNOWN)
      ]),
  )
  def test_get_children(self, sexp, node_kind, expected_result):
    sexp_graph = sexpression_graphs.SExpressionGraph(sexp)
    hol_light_graph = hol_light_sexpression_graphs.HolLightSExpressionGraph(
        sexp_graph, has_type_atoms=False)
    node = sexpression_graphs.to_node_id(sexp)
    result = hol_light_graph.get_children(node, node_kind)
    self.assertEqual(
        frozenset(expected_result),
        frozenset([(sexp_graph.to_text(child_node), child_node_kind)
                   for child_node, child_node_kind in result]))

  @parameterized.parameters(
      ('A', hol.SyntaxElementKind.TYPE, []),
      ('(<TYPE> bool)', hol.SyntaxElementKind.TYPE, [
          ('<TYPE>', hol.SyntaxElementKind.TYPE_ATOM),
          ('bool', hol.SyntaxElementKind.TYPE_CONSTRUCTOR_NAME)
      ]),
      ('(<TYPE> fun x y)', hol.SyntaxElementKind.TYPE, [
          ('<TYPE>', hol.SyntaxElementKind.TYPE_ATOM),
          ('fun', hol.SyntaxElementKind.TYPE_CONSTRUCTOR_NAME),
          ('x', hol.SyntaxElementKind.TYPE), ('y', hol.SyntaxElementKind.TYPE)
      ]),
      ('(<TYPE> bool)', hol.SyntaxElementKind.UNKNOWN, [
          ('<TYPE>', hol.SyntaxElementKind.TYPE_ATOM),
          ('bool', hol.SyntaxElementKind.TYPE_CONSTRUCTOR_NAME)
      ]),
      ('(<TYPE> fun x y)', hol.SyntaxElementKind.UNKNOWN, [
          ('<TYPE>', hol.SyntaxElementKind.TYPE_ATOM),
          ('fun', hol.SyntaxElementKind.TYPE_CONSTRUCTOR_NAME),
          ('x', hol.SyntaxElementKind.TYPE), ('y', hol.SyntaxElementKind.TYPE)
      ]),
      ('(c (<TYPE> bool) T)', hol.SyntaxElementKind.TERM, [
          ('c', hol.SyntaxElementKind.SEXPRESSION_KIND),
          ('(<TYPE> bool)', hol.SyntaxElementKind.TYPE),
          ('T', hol.SyntaxElementKind.CONSTANT_NAME)
      ]),
      ('(v (<TYPE> bool) x)', hol.SyntaxElementKind.TERM, [
          ('v', hol.SyntaxElementKind.SEXPRESSION_KIND),
          ('(<TYPE> bool)', hol.SyntaxElementKind.TYPE),
          ('x', hol.SyntaxElementKind.VARIABLE_NAME)
      ]),
      ('(l (v (<TYPE> bool) x) (v (<TYPE> bool) x))',
       hol.SyntaxElementKind.TERM, [
           ('l', hol.SyntaxElementKind.SEXPRESSION_KIND),
           ('(v (<TYPE> bool) x)', hol.SyntaxElementKind.TERM),
           ('(v (<TYPE> bool) x)', hol.SyntaxElementKind.TERM)
       ]),
      ('(a (v (<TYPE> fun (<TYPE> bool) (<TYPE> bool)) x) (v (<TYPE> bool) y))',
       hol.SyntaxElementKind.TERM, [
           ('a', hol.SyntaxElementKind.SEXPRESSION_KIND),
           ('(v (<TYPE> fun (<TYPE> bool) (<TYPE> bool)) x)',
            hol.SyntaxElementKind.TERM),
           ('(v (<TYPE> bool) y)', hol.SyntaxElementKind.TERM)
       ]),
      ('(c (<TYPE> bool) T)', hol.SyntaxElementKind.UNKNOWN, [
          ('c', hol.SyntaxElementKind.SEXPRESSION_KIND),
          ('(<TYPE> bool)', hol.SyntaxElementKind.TYPE),
          ('T', hol.SyntaxElementKind.CONSTANT_NAME)
      ]),
      ('(v (<TYPE> bool) x)', hol.SyntaxElementKind.UNKNOWN, [
          ('v', hol.SyntaxElementKind.SEXPRESSION_KIND),
          ('(<TYPE> bool)', hol.SyntaxElementKind.TYPE),
          ('x', hol.SyntaxElementKind.VARIABLE_NAME)
      ]),
      ('(l (v (<TYPE> bool) x) (v (<TYPE> bool) x))',
       hol.SyntaxElementKind.UNKNOWN, [
           ('l', hol.SyntaxElementKind.SEXPRESSION_KIND),
           ('(v (<TYPE> bool) x)', hol.SyntaxElementKind.TERM),
           ('(v (<TYPE> bool) x)', hol.SyntaxElementKind.TERM)
       ]),
      ('(a (v (<TYPE> fun (<TYPE> bool) (<TYPE> bool)) x) (v (<TYPE> bool) y))',
       hol.SyntaxElementKind.UNKNOWN, [
           ('a', hol.SyntaxElementKind.SEXPRESSION_KIND),
           ('(v (<TYPE> fun (<TYPE> bool) (<TYPE> bool)) x)',
            hol.SyntaxElementKind.TERM),
           ('(v (<TYPE> bool) y)', hol.SyntaxElementKind.TERM)
       ]),
      ('(g () (v (<TYPE> bool) x))', hol.SyntaxElementKind.UNKNOWN, [
          ('g', hol.SyntaxElementKind.UNKNOWN),
          ('()', hol.SyntaxElementKind.UNKNOWN),
          ('(v (<TYPE> bool) x)', hol.SyntaxElementKind.UNKNOWN)
      ]),
  )
  def test_get_children_with_type_atoms(self, sexp, node_kind, expected_result):
    sexp_graph = sexpression_graphs.SExpressionGraph(sexp)
    hol_light_graph = hol_light_sexpression_graphs.HolLightSExpressionGraph(
        sexp_graph, has_type_atoms=True)
    node = sexpression_graphs.to_node_id(sexp)
    result = hol_light_graph.get_children(node, node_kind)
    self.assertEqual(
        frozenset(expected_result),
        frozenset([(sexp_graph.to_text(child_node), child_node_kind)
                   for child_node, child_node_kind in result]))


if __name__ == '__main__':
  tf.test.main()
