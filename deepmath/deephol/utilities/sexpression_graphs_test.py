"""Tests for deephol.utilities.expression_graphs."""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf

from deepmath.deephol.utilities import sexpression_graphs
from deepmath.proof_assistant import proof_assistant_pb2


class SExpressionGraphsTest(parameterized.TestCase):

  def test_empty_list(self):
    g = sexpression_graphs.SExpressionGraph()
    g.add_sexp([])
    self.assertEmpty(g.labels)
    self.assertEmpty(g.parents)
    self.assertEmpty(g.children)

  def test_is_empty_string(self):
    g = sexpression_graphs.SExpressionGraph()
    self.assertFalse(g.is_empty_string())
    g.add_sexp('')
    self.assertTrue(g.is_empty_string())
    g.add_sexp('')
    self.assertTrue(g.is_empty_string())
    g.add_sexp('asdf')
    self.assertFalse(g.is_empty_string())
    g.add_sexp('')
    self.assertFalse(g.is_empty_string())

  def test_is_not_empty_string(self):
    g = sexpression_graphs.SExpressionGraph()
    g.add_sexp(['asdf', ''])
    self.assertFalse(g.is_empty_string())

  def test_init_empty_list(self):
    g = sexpression_graphs.SExpressionGraph([])
    self.assertEmpty(g.labels)
    self.assertEmpty(g.parents)
    self.assertEmpty(g.children)

  def test_none(self):
    g = sexpression_graphs.SExpressionGraph(None)
    self.assertEmpty(g.labels)
    self.assertEmpty(g.parents)
    self.assertEmpty(g.children)

  def test_get_label(self):
    g = sexpression_graphs.SExpressionGraph('(a (b a))')
    self.assertEqual(g.get_label('(a (b a))'), None)
    self.assertEqual(g.get_label('(b a)'), None)
    self.assertEqual(g.get_label('a'), 'a')
    with self.assertRaises(Exception):
      g.get_label('c')

  def test_get_parents(self):
    g = sexpression_graphs.SExpressionGraph('(a (b a))')
    self.assertEmpty(g.get_parents('(a (b a))'))
    self.assertLen(g.get_parents('(b a)'), 1)
    self.assertLen(g.get_parents('a'), 2)
    with self.assertRaises(Exception):
      g.get_parents('c')

  def test_get_children(self):
    g = sexpression_graphs.SExpressionGraph('(a (b a))')
    self.assertLen(g.get_children('(a (b a))'), 2)
    self.assertLen(g.get_children('(b a)'), 2)
    self.assertEmpty(g.get_children('a'))
    with self.assertRaises(Exception):
      g.get_children('c')

  def test_no_argument(self):
    g = sexpression_graphs.SExpressionGraph()  # should be the same as None
    self.assertEmpty(g.labels)
    self.assertEmpty(g.parents)
    self.assertEmpty(g.children)

  def test_parentheses(self):
    g = sexpression_graphs.SExpressionGraph('()')
    self.assertLen(g.labels, 1)
    self.assertLen(g.parents, 1)
    self.assertLen(g.children, 1)
    self.assertEmpty(g.get_parents('()'))
    self.assertEmpty(g.get_children('()'))

  def test_bare_words_not_accepted(self):
    with self.assertRaises(Exception):
      sexpression_graphs.SExpressionGraph('asdf asdf')

  def test_single_bare_words_accepted(self):
    g = sexpression_graphs.SExpressionGraph('asdf')
    self.assertLen(g.labels, 1)

  def test_two_sexps_not_accepted(self):
    with self.assertRaises(Exception):
      sexpression_graphs.SExpressionGraph('(asdf) ()')

  def test_single_child(self):
    g = sexpression_graphs.SExpressionGraph('(asdf)')
    self.assertLen(g.labels, 2)
    self.assertLen(g.parents, 2)
    self.assertLen(g.children, 2)
    self.assertEmpty(g.get_parents('(asdf)'))
    self.assertLen(g.get_children('(asdf)'), 1)
    self.assertEmpty(g.get_children('asdf'))
    self.assertLen(g.get_parents('asdf'), 1)

  def test_single_child_with_parens(self):
    g = sexpression_graphs.SExpressionGraph('(())')
    self.assertLen(g.labels, 2)
    self.assertLen(g.parents, 2)
    self.assertLen(g.children, 2)
    self.assertEmpty(g.get_parents('(())'))
    self.assertLen(g.get_children('(())'), 1)
    self.assertEmpty(g.get_children('()'))
    self.assertLen(g.get_parents('()'), 1)

  def test_two_children(self):
    g = sexpression_graphs.SExpressionGraph('(asdf ())')
    self.assertLen(g.labels, 3)
    self.assertLen(g.parents, 3)
    self.assertLen(g.children, 3)
    self.assertEmpty(g.get_parents('(asdf ())'))
    self.assertLen(g.get_children('(asdf ())'), 2)
    self.assertEmpty(g.get_children('asdf'))
    self.assertLen(g.get_parents('asdf'), 1)
    self.assertEmpty(g.get_children('()'))
    self.assertLen(g.get_parents('()'), 1)

  def test_two_equal_children(self):
    g = sexpression_graphs.SExpressionGraph('(asdf asdf)')
    self.assertLen(g.labels, 2)
    self.assertLen(g.parents, 2)
    self.assertLen(g.children, 2)
    self.assertEmpty(g.get_parents('(asdf asdf)'))
    self.assertLen(g.get_children('(asdf asdf)'), 2)  # order is important here
    self.assertEmpty(g.get_children('asdf'))
    self.assertLen(g.get_parents('asdf'), 1)  # order of parents not important

  def test_variable_named_v(self):
    g = sexpression_graphs.SExpressionGraph('(v N v)')
    self.assertLen(g.labels, 3)
    self.assertLen(g.parents, 3)
    self.assertLen(g.children, 3)
    self.assertEmpty(g.get_parents('(v N v)'))
    self.assertLen(g.get_parents('v'), 1)  # parents pointers collapsed
    self.assertLen(g.get_children('(v N v)'), 3)  # order is important here
    self.assertTrue(g.is_leaf_node(sexpression_graphs.to_node_id('v')))

  def test_real_expression(self):
    """Test parsing expressions taken from the HOL Light theorem prover.

    The string below is a statement on some property of sets taken from the HOL
    Light theorem prover in //third_party/hol_light. The test checks if we can
    parse a 'typical' expression from the data source this parser is targeted
    for.
    """
    g = sexpression_graphs.SExpressionGraph(
        '(a (c (fun (fun (fun (fun (prod ?0 ?0) (bool)) (bool)) (bool)) '
        '(bool)) !) (l (v (fun (fun (prod ?0 ?0) (bool)) (bool)) P) (a (c (fun '
        '(fun (fun (prod A A) (bool)) (bool)) (bool)) !) (l (v (fun (prod A A) '
        '(bool)) l) (a (a (c (fun (bool) (fun (bool) (bool))) =) (a (a (c (fun '
        '(fun (prod ?0 ?0) (bool)) (fun ?0 (bool))) fl) (a (c (fun (fun (fun '
        '(prod ?0 ?0) (bool)) (bool)) (fun (prod ?0 ?0) (bool))) UNIONS) (v '
        '(fun (fun (prod ?0 ?0) (bool)) (bool)) P))) (v ?0 x))) (a (c (fun '
        '(fun (fun (prod ?0 ?0) (bool)) (bool)) (bool)) ?) (l (v (fun (prod ?0 '
        '?0) (bool)) l) (a (a (c (fun (bool) (fun (bool) (bool))) /\\) (a (v '
        '(fun (fun (prod ?0 ?0) (bool)) (bool)) P) (v (fun (prod ?0 ?0) '
        '(bool)) l))) (a (a (c (fun (fun (prod ?0 ?0) (bool)) (fun ?0 (bool))) '
        'fl) (v (fun (prod ?0 ?0) (bool)) l)) (v ?0 x))))))))))')
    self.assertLen(g.labels, 59)

  def test_repeated_add(self):
    g = sexpression_graphs.SExpressionGraph(
        '(v (fun (fun (prod ?0 ?0) (bool)) (bool)) P)')
    self.assertLen(g.labels, 11)
    g.add_sexp('(fun (fun (prod ?0 ?0) (bool)) (bool))')
    self.assertLen(g.labels, 11)
    g.add_sexp('(asdf)')
    self.assertLen(g.labels, 13)

  def test_add_list(self):
    g = sexpression_graphs.SExpressionGraph()
    g.add_sexp([
        '(v (fun (fun (prod ?0 ?0) (bool)) (bool)) P)',
        '(fun (fun (prod ?0 ?0) (bool)) (bool))'
    ])
    self.assertLen(g.labels, 11)
    g.add_sexp('(asdf)')
    self.assertLen(g.labels, 13)

  @parameterized.named_parameters(
      ('theorem',
       proof_assistant_pb2.Theorem(
           conclusion='(v (fun (prod ?0 ?0) (bool)) P)',
           hypotheses=['(v (fun (prod ?0 ?0) (bool)) Q)'],
           tag=proof_assistant_pb2.Theorem.THEOREM),
       '(h ((v (fun (prod ?0 ?0) (bool)) Q)) (v (fun (prod ?0 ?0) (bool)) P))'),
      ('goal',
       proof_assistant_pb2.Theorem(
           conclusion='(v (fun (prod ?0 ?0) (bool)) P)',
           hypotheses=['(v (fun (prod ?0 ?0) (bool)) Q)'],
           tag=proof_assistant_pb2.Theorem.GOAL),
       '(g ((v (fun (prod ?0 ?0) (bool)) Q)) (v (fun (prod ?0 ?0) (bool)) P))'),
      ('definition',
       proof_assistant_pb2.Theorem(
           conclusion='(v (fun (prod ?0 ?0) (bool)) P)',
           definition=proof_assistant_pb2.Definition(
               constants=['constant1', 'constant2']),
           tag=proof_assistant_pb2.Theorem.DEFINITION),
       '(d (constant1 constant2) (v (fun (prod ?0 ?0) (bool)) P))'),
      ('type_definition',
       proof_assistant_pb2.Theorem(
           conclusion='(v (fun (prod ?0 ?0) (bool)) P)',
           type_definition=proof_assistant_pb2.TypeDefinition(
               type_name='test_type_name'),
           tag=proof_assistant_pb2.Theorem.TYPE_DEFINITION),
       '(t test_type_name (v (fun (prod ?0 ?0) (bool)) P))'))
  def test_add_theorem(self, theorem, expected_val):

    g = sexpression_graphs.SExpressionGraph(theorem)
    self.assertLen(g.roots(), 1)
    self.assertEqual(g.to_text(g.roots()[0]), expected_val)

  def test_multiple_expressions(self):
    """A canary for interpretation changes and performance problems.

    The expressions here were taken from the HOL Light theorem prover, the
    anticipated source of data for this parser. See also //third_party/hol_light

    This test checks for the number of graph nodes after parsing and
    deduplicating subexpressions.
    """
    g = sexpression_graphs.SExpressionGraph()
    g.add_sexp(
        '(a (a (c (fun (bool) (fun (bool) (bool))) ==>) (a (c (fun (fun ?0 '
        '(bool)) (bool)) ?) (l (v ?0 a) (a (a (c (fun (fun ?0 (bool)) (fun '
        '(fun ?0 (bool)) (bool))) SUBSET) (v (fun ?0 (bool)) s)) (a (a (c (fun '
        '?0 (fun (fun ?0 (bool)) (fun ?0 (bool)))) INSERT) (v ?0 a)) (c (fun '
        '?0 (bool)) EMPTY)))))) (a (c (fun (fun ?0 (bool)) (bool)) FINITE) (v '
        '(fun ?0 (bool)) s)))')
    g.add_sexp(
        '(a (a (c (fun (fun ?0 (bool)) (fun (fun ?0 (bool)) (bool))) =) (a (c '
        '(fun (fun ?0 (bool)) (fun ?0 (bool))) GSPEC) (l (v ?0 GEN%PVAR%0) (a '
        '(c (fun (fun ?0 (bool)) (bool)) ?) (l (v ?0 x) (a (a (a (c (fun ?0 '
        '(fun (bool) (fun ?0 (bool)))) SETSPEC) (v ?0 GEN%PVAR%0)) (a (a (c '
        '(fun (bool) (fun (bool) (bool))) /\\) (a (v (fun ?0 (bool)) P) (v '
        '?0 x))) (a (c (fun (bool) (bool)) ~) (a (v (fun ?0 (bool)) Q) (v ?0 '
        'x))))) (v ?0 x))))))) (a (a (c (fun (fun ?0 (bool)) (fun (fun ?0 '
        '(bool)) (fun ?0 (bool)))) DIFF) (a (c (fun (fun ?0 (bool)) (fun ?0 '
        '(bool))) GSPEC) (l (v ?0 GEN%PVAR%0) (a (c (fun (fun ?0 (bool)) '
        '(bool)) ?) (l (v ?0 x) (a (a (a (c (fun ?0 (fun (bool) (fun ?0 '
        '(bool)))) SETSPEC) (v ?0 GEN%PVAR%0)) (a (v (fun ?0 (bool)) P) '
        '(v ?0 x))) (v ?0 x))))))) (a (c (fun (fun ?0 (bool)) (fun ?0 '
        '(bool))) GSPEC) (l (v ?0 GEN%PVAR%0) (a (c (fun (fun ?0 (bool)) '
        '(bool)) ?) (l (v ?0 x) (a (a (a (c (fun ?0 (fun (bool) (fun ?0 '
        '(bool)))) SETSPEC) (v ?0 GEN%PVAR%0)) (a (a (c (fun (bool) (fun '
        '(bool) (bool))) /\\) (a (v (fun ?0 (bool)) P) (v ?0 x))) (a (v '
        '(fun ?0 (bool)) Q) (v ?0 x)))) (v ?0 x))))))))')
    g.add_sexp(
        '(a (c (fun (fun (fun (cart (real) M) (cart (real) N)) (bool)) (bool)) '
        '!) (l (v (fun (cart (real) M) (cart (real) N)) f) (a (c (fun (fun '
        '(fun (cart (real) M) (bool)) (bool)) (bool)) !) (l (v (fun (cart '
        '(real) M) (bool)) s) (a (c (fun (fun (fun (cart (real) M) (bool)) '
        '(bool)) (bool)) !) (l (v (fun (cart (real) M) (bool)) u) (a (c (fun '
        '(fun (cart (real) M) (bool)) (bool)) !) (l (v (cart (real) M) x) (a '
        '(a (c (fun (bool) (fun (bool) (bool))) ==>) (a (a (c (fun (bool) (fun '
        '(bool) (bool))) /\\) (a (a (c (fun (cart (real) M) (fun (fun (cart '
        '(real) M) (bool)) (bool))) limit_point_of) (v (cart (real) M) x)) (v '
        '(fun (cart (real) M) (bool)) s))) (a (a (c (fun (bool) (fun (bool) '
        '(bool))) /\\) (a (a (c (fun (fun (cart (real) M) (cart (real) N)) '
        '(fun (net (cart (real) M)) (bool))) continuous) (v (fun (cart (real) '
        'M) (cart (real) N)) f)) (a (a (c (fun (net (cart (real) M)) (fun '
        '(fun (cart (real) M) (bool)) (net (cart (real) M)))) within) (a (c '
        '(fun (cart (real) M) (net (cart (real) M))) at) (v (cart (real) M) '
        'x))) (v (fun (cart (real) M) (bool)) s)))) (a (a (c (fun (bool) '
        '(fun (bool) (bool))) /\\) (a (c (fun (fun (cart (real) M) (bool)) '
        '(bool)) open) (v (fun (cart (real) M) (bool)) u))) (a (a (c (fun '
        '(bool) (fun (bool) (bool))) /\\) (a (a (c (fun (cart (real) M) (fun '
        '(fun (cart (real) M) (bool)) (bool))) IN) (v (cart (real) M) x)) (v '
        '(fun (cart (real) M) (bool)) u))) (a (c (fun (fun (cart (real) M) '
        '(bool)) (bool)) FINITE) (a (c (fun (fun (cart (real) M) (bool)) (fun '
        '(cart (real) M) (bool))) GSPEC) (l (v (cart (real) M) GEN%PVAR%0) (a '
        '(c (fun (fun (cart (real) M) (bool)) (bool)) ?) (l (v (cart (real) M) '
        'z) (a (a (a (c (fun (cart (real) M) (fun (bool) (fun (cart (real) M) '
        '(bool)))) SETSPEC) (v (cart (real) M) GEN%PVAR%0)) (a (a (c (fun '
        '(bool) (fun (bool) (bool))) /\\) (a (a (c (fun (cart (real) M) (fun '
        '(fun (cart (real) M) (bool)) (bool))) IN) (v (cart (real) M) z)) (a '
        '(a (c (fun (fun (cart (real) M) (bool)) (fun (fun (cart (real) M) '
        '(bool)) (fun (cart (real) M) (bool)))) INTER) (v (fun (cart (real) M) '
        '(bool)) s)) (v (fun (cart (real) M) (bool)) u)))) (a (a (c (fun (cart '
        '(real) N) (fun (cart (real) N) (bool))) =) (a (v (fun (cart (real) M) '
        '(cart (real) N)) f) (v (cart (real) M) z))) (a (v (fun (cart (real) M)'
        ' (cart (real) N)) f) (v (cart (real) M) x))))) (v (cart (real) M) '
        'z)))))))))))) (a (a (c (fun (cart (real) N) (fun (fun (cart (real) '
        'N) (bool)) (bool))) limit_point_of) (a (v (fun (cart (real) M) (cart '
        '(real) N)) f) (v (cart (real) M) x))) (a (a (c (fun (fun (cart (real) '
        'M) (cart (real) N)) (fun (fun (cart (real) M) (bool)) (fun (cart '
        '(real) N) (bool)))) IMAGE) (v (fun (cart (real) M) (cart (real) N)) '
        'f)) (v (fun (cart (real) M) (bool)) s))))))))))))')
    g.add_sexp(
        '(a (c (fun (fun (fun (cart (real) M) (cart (real) N)) (bool)) (bool)) '
        '!) (l (v (fun (cart (real) M) (cart (real) N)) f) (a (c (fun (fun '
        '(fun (cart (real) M) (bool)) (bool)) (bool)) !) (l (v (fun (cart '
        '(real) M) (bool)) s) (a (c (fun (fun (cart (real) M) (bool)) (bool)) '
        '!) (l (v (cart (real) M) x) (a (a (c (fun (bool) (fun (bool) (bool))) '
        '==>) (a (a (c (fun (bool) (fun (bool) (bool))) /\\) (a (a (c (fun '
        '(cart (real) M) (fun (fun (cart (real) M) (bool)) (bool))) '
        'limit_point_of) (v (cart (real) M) x)) (v (fun (cart (real) M) '
        '(bool)) s))) (a (a (c (fun (bool) (fun (bool) (bool))) /\\) (a (a '
        '(c (fun (fun (cart (real) M) (cart (real) N)) (fun (net (cart (real) '
        'M)) (bool))) continuous) (v (fun (cart (real) M) (cart (real) N)) '
        'f)) (a (a (c (fun (net (cart (real) M)) (fun (fun (cart (real) M) '
        '(bool)) (net (cart (real) M)))) within) (a (c (fun (cart (real) M) '
        '(net (cart (real) M))) at) (v (cart (real) M) x))) (v (fun (cart '
        '(real) M) (bool)) s)))) (a (c (fun (fun (cart (real) M) (bool)) '
        '(bool)) !) (l (v (cart (real) M) x) (a (c (fun (fun (cart (real) M) '
        '(bool)) (bool)) !) (l (v (cart (real) M) y) (a (a (c (fun (bool) (fun '
        '(bool) (bool))) ==>) (a (a (c (fun (bool) (fun (bool) (bool))) /\\) '
        '(a (a (c (fun (cart (real) M) (fun (fun (cart (real) M) (bool)) '
        '(bool))) IN) (v (cart (real) M) x)) (v (fun (cart (real) M) (bool)) '
        's))) (a (a (c (fun (bool) (fun (bool) (bool))) /\\) (a (a (c (fun '
        '(cart (real) M) (fun (fun (cart (real) M) (bool)) (bool))) IN) (v '
        '(cart (real) M) y)) (v (fun (cart (real) M) (bool)) s))) (a (a (c '
        '(fun (cart (real) N) (fun (cart (real) N) (bool))) =) (a (v (fun '
        '(cart (real) M) (cart (real) N)) f) (v (cart (real) M) x))) (a (v '
        '(fun (cart (real) M) (cart (real) N)) f) (v (cart (real) M) y)))))) '
        '(a (a (c (fun (cart (real) M) (fun (cart (real) M) (bool))) =) (v '
        '(cart (real) M) x)) (v (cart (real) M) y)))))))))) (a (a (c (fun '
        '(cart (real) N) (fun (fun (cart (real) N) (bool)) (bool))) '
        'limit_point_of) (a (v (fun (cart (real) M) (cart (real) N)) f) '
        '(v (cart (real) M) x))) (a (a (c (fun (fun (cart (real) M) (cart '
        '(real) N)) (fun (fun (cart (real) M) (bool)) (fun (cart (real) N) '
        '(bool)))) IMAGE) (v (fun (cart (real) M) (cart (real) N)) f)) (v '
        '(fun (cart (real) M) (bool)) s))))))))))')
    g.add_sexp(
        '(a (c (fun (fun (fun ?0 (bool)) (bool)) (bool)) !) (l (v (fun ?0 '
        '(bool)) s) (a (a (c (fun (bool) (fun (bool) (bool))) ==>) (a (a (c '
        '(fun (bool) (fun (bool) (bool))) /\\) (a (a (c (fun (fun ?0 (bool)) '
        '(fun (fun ?0 (bool)) (bool))) SUBSET) (v (fun ?0 (bool)) s)) (v (fun '
        '?0 (bool)) t))) (a (a (c (fun ?0 (fun (fun ?0 (bool)) (bool))) IN) (v '
        '?0 x)) (v (fun ?0 (bool)) s)))) (a (a (c (fun ?0 (fun (fun ?0 (bool)) '
        '(bool))) IN) (v ?0 x)) (v (fun ?0 (bool)) t)))))')
    g.add_sexp(
        '(a (c (fun (fun (fun (cart (real) M) (cart (real) N)) (bool)) (bool)) '
        '!) (l (v (fun (cart (real) M) (cart (real) N)) f) (a (c (fun (fun '
        '(fun (num) (fun (cart (real) M) (bool))) (bool)) (bool)) !) (l (v '
        '(fun (num) (fun (cart (real) M) (bool))) s) (a (a (c (fun (bool) '
        '(fun (bool) (bool))) ==>) (a (a (c (fun (bool) (fun (bool) (bool))) '
        '/\\) (a (a (c (fun (fun (cart (real) M) (cart (real) N)) (fun (fun '
        '(cart (real) M) (bool)) (bool))) continuous_on) (v (fun (cart (real) '
        'M) (cart (real) N)) f)) (a (v (fun (num) (fun (cart (real) M) '
        '(bool))) s) (a (c (fun (num) (num)) NUMERAL) (c (num) _0))))) (a (a '
        '(c (fun (bool) (fun (bool) (bool))) /\\) (a (c (fun (fun (num) '
        '(bool)) (bool)) !) (l (v (num) n) (a (c (fun (fun (cart (real) M) '
        '(bool)) (bool)) compact) (a (v (fun (num) (fun (cart (real) M) '
        '(bool))) s) (v (num) n)))))) (a (c (fun (fun (num) (bool)) (bool)) '
        '!) (l (v (num) n) (a (a (c (fun (fun (cart (real) M) (bool)) (fun '
        '(fun (cart (real) M) (bool)) (bool))) SUBSET) (a (v (fun (num) '
        '(fun (cart (real) M) (bool))) s) (a (c (fun (num) (num)) SUC) (v '
        '(num) n)))) (a (v (fun (num) (fun (cart (real) M) (bool))) s) (v ('
        'num) n)))))))) (a (a (c (fun (fun (cart (real) N) (bool)) (fun (fu'
        'n (cart (real) N) (bool)) (bool))) =) (a (a (c (fun (fun (cart (re'
        'al) M) (cart (real) N)) (fun (fun (cart (real) M) (bool)) (fun (ca'
        'rt (real) N) (bool)))) IMAGE) (v (fun (cart (real) M) (cart (real) '
        'N)) f)) (a (c (fun (fun (fun (cart (real) M) (bool)) (bool)) (fun '
        '(cart (real) M) (bool))) INTERS) (a (c (fun (fun (fun (cart (real) '
        'M) (bool)) (bool)) (fun (fun (cart (real) M) (bool)) (bool))) GSPEC'
        ') (l (v (fun (cart (real) M) (bool)) GEN%PVAR%0) (a (c (fun (fun (n'
        'um) (bool)) (bool)) ?) (l (v (num) n) (a (a (a (c (fun (fun (cart ('
        'real) M) (bool)) (fun (bool) (fun (fun (cart (real) M) (bool)) (boo'
        'l)))) SETSPEC) (v (fun (cart (real) M) (bool)) GEN%PVAR%0)) (a (a '
        '(c (fun (num) (fun (fun (num) (bool)) (bool))) IN) (v (num) n)) (c '
        '(fun (num) (bool)) UNIV))) (a (v (fun (num) (fun (cart (real) M) '
        '(bool))) s) (v (num) n)))))))))) (a (c (fun (fun (fun (cart (real) '
        'N) (bool)) (bool)) (fun (cart (real) N) (bool))) INTERS) (a (c ('
        'fun (fun (fun (cart (real) N) (bool)) (bool)) (fun (fun (cart (real) '
        'N) (bool)) (bool))) GSPEC) (l (v (fun (cart (real) N) (bool)) '
        'GEN%PVAR%0) (a (c (fun (fun (num) (bool)) (bool)) ?) (l (v (num) '
        'n) (a (a (a (c (fun (fun (cart (real) N) (bool)) (fun (bool) (fun '
        '(fun (cart (real) N) (bool)) (bool)))) SETSPEC) (v (fun (cart (real'
        ') N) (bool)) GEN%PVAR%0)) (a (a (c (fun (num) (fun (fun (num) (bool'
        ')) (bool))) IN) (v (num) n)) (c (fun (num) (bool)) UNIV))) (a (a (c '
        '(fun (fun (cart (real) M) (cart (real) N)) (fun (fun (cart (real) M) '
        '(bool)) (fun (cart (real) N) (bool)))) IMAGE) (v (fun (cart (real) M) '
        '(cart (real) N)) f)) (a (v (fun (num) (fun (cart (real) M) (bool))) '
        's) (v (num) n)))))))))))))))')
    g.add_sexp(
        '(a (c (fun (fun (fun (cart (real) M) (cart (real) N)) (bool)) (bool)) '
        '!) (l (v (fun (cart (real) M) (cart (real) N)) f) (a (c (fun (fun ('
        'fun (num) (fun (cart (real) M) (bool))) (bool)) (bool)) !) (l (v ('
        'fun (num) (fun (cart (real) M) (bool))) s) (a (c (fun (fun (num) (boo'
        'l)) (bool)) !) (l (v (num) m) (a (a (c (fun (bool) (fun (bool) (bool'
        '))) ==>) (a (a (c (fun (bool) (fun (bool) (bool))) /\\) (a (a (c (fu'
        'n (fun (cart (real) M) (cart (real) N)) (fun (fun (cart (real) M) (b'
        'ool)) (bool))) continuous_on) (v (fun (cart (real) M) (cart (real) '
        'N)) f)) (a (v (fun (num) (fun (cart (real) M) (bool))) s) (v (num) '
        'm)))) (a (a (c (fun (bool) (fun (bool) (bool))) /\\) (a (c (fun ('
        'fun (num) (bool)) (bool)) !) (l (v (num) n) (a (a (c (fun (bool) ('
        'fun (bool) (bool))) ==>) (a (a (c (fun (num) (fun (num) (bool))) '
        '<=) (v (num) m)) (v (num) n))) (a (c (fun (fun (cart (real) M) '
        '(bool)) (bool)) compact) (a (v (fun (num) (fun (cart (real) M) (bool'
        '))) s) (v (num) n))))))) (a (c (fun (fun (num) (bool)) (bool)) !) ('
        'l (v (num) n) (a (a (c (fun (bool) (fun (bool) (bool))) ==>) (a (a '
        '(c (fun (num) (fun (num) (bool))) <=) (v (num) m)) (v (num) n))) (a '
        '(a (c (fun (fun (cart (real) M) (bool)) (fun (fun (cart (real) M) '
        '(bool)) (bool))) SUBSET) (a (v (fun (num) (fun (cart (real) M) (bool'
        '))) s) (a (c (fun (num) (num)) SUC) (v (num) n)))) (a (v (fun (num) '
        '(fun (cart (real) M) (bool))) s) (v (num) n))))))))) (a (a (c (fun ('
        'fun (cart (real) N) (bool)) (fun (fun (cart (real) N) (bool)) (bool)'
        ')) =) (a (a (c (fun (fun (cart (real) M) (cart (real) N)) (fun (fun '
        '(cart (real) M) (bool)) (fun (cart (real) N) (bool)))) IMAGE) (v (fun '
        '(cart (real) M) (cart (real) N)) f)) (a (c (fun (fun (fun (cart (real'
        ') M) (bool)) (bool)) (fun (cart (real) M) (bool))) INTERS) (a (c ('
        'fun (fun (fun (cart (real) M) (bool)) (bool)) (fun (fun (cart (real) '
        'M) (bool)) (bool))) GSPEC) (l (v (fun (cart (real) M) (bool)) GEN%PVA'
        'R%0) (a (c (fun (fun (num) (bool)) (bool)) ?) (l (v (num) n) (a (a (a '
        '(c (fun (fun (cart (real) M) (bool)) (fun (bool) (fun (fun (cart (real'
        ') M) (bool)) (bool)))) SETSPEC) (v (fun (cart (real) M) (bool)) GEN%PV'
        'AR%0)) (a (a (c (fun (num) (fun (num) (bool))) <=) (v (num) m)) (v (nu'
        'm) n))) (a (v (fun (num) (fun (cart (real) M) (bool))) s) (v (num) n))'
        ')))))))) (a (c (fun (fun (fun (cart (real) N) (bool)) (bool)) (fun (c'
        'art (real) N) (bool))) INTERS) (a (c (fun (fun (fun (cart (real) N) (b'
        'ool)) (bool)) (fun (fun (cart (real) N) (bool)) (bool))) GSPEC) (l (v '
        '(fun (cart (real) N) (bool)) GEN%PVAR%0) (a (c (fun (fun (num) (bool)'
        ') (bool)) ?) (l (v (num) n) (a (a (a (c (fun (fun (cart (real) N) (boo'
        'l)) (fun (bool) (fun (fun (cart (real) N) (bool)) (bool)))) SETSPEC) '
        '(v (fun (cart (real) N) (bool)) GEN%PVAR%0)) (a (a (c (fun (num) (fun '
        '(num) (bool))) <=) (v (num) m)) (v (num) n))) (a (a (c (fun (fun '
        '(cart (real) M) (cart (real) N)) (fun (fun (cart (real) M) (bool)) ('
        'fun (cart (real) N) (bool)))) IMAGE) (v (fun (cart (real) M) (cart ('
        'real) N)) f)) (a (v (fun (num) (fun (cart (real) M) (bool))) s) (v (n'
        'um) n)))))))))))))))))')
    g.add_sexp(
        '(a (c (fun (fun (net ?0) (bool)) (bool)) !) (l (v (net ?0) net) (a (c '
        '(fun (fun (cart (real) ?1) (bool)) (bool)) !) (l (v (cart (real) ?1) '
        'c) (a (a (c (fun (fun ?0 (cart (real) ?1)) (fun (net ?0) (bool))) cont'
        'inuous) (l (v ?0 x) (v (cart (real) ?1) c))) (v (net ?0) net))))))')
    g.add_sexp(
        '(a (c (fun (fun (fun ?0 (cart (real) ?1)) (bool)) (bool)) !) (l (v '
        '(fun ?0 (cart (real) ?1)) f) (a (c (fun (fun (real) (bool)) (bool)) '
        '!) (l (v (real) c) (a (c (fun (fun (net ?0) (bool)) (bool)) !) (l (v '
        '(net ?0) net) (a (a (c (fun (bool) (fun (bool) (bool))) ==>) (a (a (c '
        '(fun (fun ?0 (cart (real) ?1)) (fun (net ?0) (bool))) continuous) (v ('
        'fun ?0 (cart (real) ?1)) f)) (v (net ?0) net))) (a (a (c (fun (fun ?0 '
        '(cart (real) ?1)) (fun (net ?0) (bool))) continuous) (l (v ?0 x) (a ('
        'a (c (fun (real) (fun (cart (real) ?1) (cart (real) ?1))) %) (v (real'
        ') c)) (a (v (fun ?0 (cart (real) ?1)) f) (v ?0 x))))) (v (net ?0) net)'
        '))))))))')
    g.add_sexp(
        '(a (c (fun (fun (fun ?0 (cart (real) ?1)) (bool)) (bool)) !) (l (v (fu'
        'n ?0 (cart (real) ?1)) f) (a (c (fun (fun (net ?0) (bool)) (bool)) !) '
        '(l (v (net ?0) net) (a (a (c (fun (bool) (fun (bool) (bool))) ==>) (a '
        '(a (c (fun (fun ?0 (cart (real) ?1)) (fun (net ?0) (bool))) continuous'
        ') (v (fun ?0 (cart (real) ?1)) f)) (v (net ?0) net))) (a (a (c (fun (f'
        'un ?0 (cart (real) ?1)) (fun (net ?0) (bool))) continuous) (l (v ?0 x'
        ') (a (c (fun (cart (real) ?1) (cart (real) ?1)) vector_neg) (a (v (fu'
        'n ?0 (cart (real) ?1)) f) (v ?0 x))))) (v (net ?0) net)))))))')
    g.add_sexp(
        '(a (c (fun (fun (fun ?0 (cart (real) ?1)) (bool)) (bool)) !) (l (v (fu'
        'n ?0 (cart (real) ?1)) f) (a (c (fun (fun (fun ?0 (cart (real) ?1)) (b'
        'ool)) (bool)) !) (l (v (fun ?0 (cart (real) ?1)) g) (a (c (fun (fun (n'
        'et ?0) (bool)) (bool)) !) (l (v (net ?0) net) (a (a (c (fun (bool) (fu'
        'n (bool) (bool))) ==>) (a (a (c (fun (bool) (fun (bool) (bool))) /\\) '
        '(a (a (c (fun (fun ?0 (cart (real) ?1)) (fun (net ?0) (bool))) continu'
        'ous) (v (fun ?0 (cart (real) ?1)) f)) (v (net ?0) net))) (a (a (c (fun'
        ' (fun ?0 (cart (real) ?1)) (fun (net ?0) (bool))) continuous) (v (fun '
        '?0 (cart (real) ?1)) g)) (v (net ?0) net)))) (a (a (c (fun (fun ?0 (ca'
        'rt (real) ?1)) (fun (net ?0) (bool))) continuous) (l (v ?0 x) (a (a (c'
        ' (fun (cart (real) ?1) (fun (cart (real) ?1) (cart (real) ?1))) vector'
        '_add) (a (v (fun ?0 (cart (real) ?1)) f) (v ?0 x))) (a (v (fun ?0 (car'
        't (real) ?1)) g) (v ?0 x))))) (v (net ?0) net)))))))))')
    self.assertLen(g.labels, 479)

  @parameterized.named_parameters(
      ('empty_string', ''),
      ('single_word_2', 'asdf'),
      ('single_word_with_specials', 'trololo!@#$%^&*_-"\';:+={}[]|.,?~`\\lo'),
      ('single_word_malformed_escapes', 'trololo!@#$%^&*_-"\';:+={}[]|.,?~`\\/'
       '\\//\\\/\nlo'),  # pylint: disable=anomalous-backslash-in-string
      ('plain_parens', '()'),
      ('singleton', '(a)'),
      ('simple_expression', '(a b)'),
      ('nontrivial_sexp', '(a (a bool c) (a bool c))'),
      ('real_expression', '(l (v (cart (real) M) y) (a (a (c (fun (bool) (fun '
       '(bool) (bool))) ==>) (a (a (c (fun (bool) (fun (bool) (bool))) /\\) '
       '(a (a (c (fun (cart (real) M) (fun (fun (cart (real) M) (bool)) '
       '(bool))) IN) (v (cart (real) M) x)) (v (fun (cart (real) M) (bool)) '
       's))) (a (a (c (fun (bool) (fun (bool) (bool))) /\\) (a (a (c (fun '
       '(cart (real) M) (fun (fun (cart (real) M) (bool)) (bool))) IN) (v '
       '(cart (real) M) y)) (v (fun (cart (real) M) (bool)) s))) (a (a (c '
       '(fun (cart (real) N) (fun (cart (real) N) (bool))) =) (a (v (fun '
       '(cart (real) M) (cart (real) N)) f) (v (cart (real) M) x))) (a (v '
       '(fun (cart (real) M) (cart (real) N)) f) (v (cart (real) M) y)))))) '
       '(a (a (c (fun (cart (real) M) (fun (cart (real) M) (bool))) =) (v '
       '(cart (real) M) x)) (v (cart (real) M) y))))'),
      ('previous_bug',
       '(a (a (c (fun (bool) (fun (bool) (bool))) /\) (a (a (c (fun (cart (real) N) (fun (fun (cart (real) N) (bool)) (bool))) IN) (v (cart (real) N) u)) (a (a (c (fun (fun (cart (real) N) (bool)) (fun (fun (cart (real) N) (bool)) (fun (cart (real) N) (bool)))) INTER) (a (c (fun (prod (cart (real) N) (real)) (fun (cart (real) N) (bool))) ball) (a (a (c (fun (cart (real) N) (fun (real) (prod (cart (real) N) (real)))) ,) (v (cart (real) N) a)) (v (real) r)))) (v (fun (cart (real) N) (bool)) t)))) (a (a (c (fun (cart (real) N) (fun (fun (cart (real) N) (bool)) (bool))) IN) (v (cart (real) N) v)) (a (a (c (fun (fun (cart (real) N) (bool)) (fun (fun (cart (real) N) (bool)) (fun (cart (real) N) (bool)))) INTER) (a (c (fun (prod (cart (real) N) (real)) (fun (cart (real) N) (bool))) ball) (a (a (c (fun (cart (real) N) (fun (real) (prod (cart (real) N) (real)))) ,) (v (cart (real) N) a)) (v (real) r)))) (v (fun (cart (real) N) (bool)) t))))'
      ))
  def test_to_text_reconstruction(self, sexp):
    """Check if expressions are the same after parsing and printing to str."""
    g = sexpression_graphs.SExpressionGraph(sexp)
    node_id = sexpression_graphs.to_node_id(sexp)
    self.assertEqual(g.to_text(node_id), sexp)

  def test_to_text_subexpressions(self):
    """Check if subexpressions can be retrieved."""
    g = sexpression_graphs.SExpressionGraph('(fun (prod A A) (bool))')
    subexpression = '(prod A A)'
    subexp_id = sexpression_graphs.to_node_id(subexpression)
    self.assertEqual(g.to_text(subexp_id), subexpression)

  def test_empty_post_order(self):
    o = sexpression_graphs.SExpressionGraph().global_post_order()
    self.assertEmpty(o)

  @parameterized.parameters(
      ('(v (fun (fun (prod ?0 ?0) (bool)) (bool)) P)',),
      ('',),
      ('(c (fun (bool) (fun (bool) (bool))) /\\)',),
      ('(a (c (fun (prod (cart (real) N) (real)) (fun (cart (real) N) (bool))) '
       'ball) (a (a (c (fun (cart (real) N) (fun (real) (prod (cart (real) N) '
       '(real)))) ,) (v (cart (real) N) a)) (v (real) r)))',),
      ([
          '(v (fun (fun (prod ?0 ?0) (bool)) (bool)) P)', '', '',
          '(v (fun (fun (prod ?0 ?0) (bool)) (bool)) Q)'
      ],),
  )
  def test_post_order_single_expr(self, sexp):
    g = sexpression_graphs.SExpressionGraph(sexp)
    o = g.global_post_order()
    self.assertLen(o, len(g))
    self.assertCountEqual(o.values(), range(len(g)))
    for n in g.nodes:
      for c in g.children[n]:
        self.assertGreater(o[n], o[c])
    roots = g.roots()
    roots.sort()
    root_pairs = zip(roots[:-1], roots[1:])
    for r1, r2 in root_pairs:
      self.assertGreater(o[r2], o[r1])

  def test_post_order_skip_first_child(self):
    g = sexpression_graphs.SExpressionGraph(
        '(c (fun (bool) (fun (bool) (bool))) /\\)')
    o = g.global_post_order(skip_first_child=True)
    self.assertLen(o, 5)
    self.assertNotIn(sexpression_graphs.to_node_id('bool'),
                     o)  # ... only occurs as '(bool)'
    self.assertIn(sexpression_graphs.to_node_id('(bool)'), o)


if __name__ == '__main__':
  tf.test.main()
