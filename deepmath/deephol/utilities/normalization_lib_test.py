"""Tests for deepmath.deephol.normalization_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf
from deepmath.deephol.utilities import normalization_lib
from deepmath.proof_assistant import proof_assistant_pb2


class NormalizedFingerprintLibTest(parameterized.TestCase):

  def test_normalize_trivial(self):
    theorem = proof_assistant_pb2.Theorem(conclusion='(does not contain types)')
    self.assertEqual(normalization_lib.normalize(theorem), theorem)

  def test_normalize_single_type(self):
    theorem = proof_assistant_pb2.Theorem(
        conclusion='(does contain type ?1234)')
    expected = proof_assistant_pb2.Theorem(conclusion='(does contain type ?0)')
    self.assertEqual(normalization_lib.normalize(theorem), expected)

  def test_normalize_multiple_occurrences(self):
    theorem = proof_assistant_pb2.Theorem(
        conclusion='(does contain types ?1234 and ?1234)')
    expected = proof_assistant_pb2.Theorem(
        conclusion='(does contain types ?0 and ?0)')
    self.assertEqual(normalization_lib.normalize(theorem), expected)

  def test_normalize_multiple_types(self):
    theorem = proof_assistant_pb2.Theorem(
        conclusion='(does contain types ?1234 and ?34598734958)')
    expected = proof_assistant_pb2.Theorem(
        conclusion='(does contain types ?0 and ?1)')
    self.assertEqual(normalization_lib.normalize(theorem), expected)

  def test_normalize_multiple_types_flipped(self):
    theorem = proof_assistant_pb2.Theorem(
        conclusion='(does contain types ?34598734958 and ?1234)')
    expected = proof_assistant_pb2.Theorem(
        conclusion='(does contain types ?0 and ?1)')
    self.assertEqual(normalization_lib.normalize(theorem), expected)

  def test_normalize_with_hypotheses(self):
    theorem = proof_assistant_pb2.Theorem(
        conclusion='(does contain types ?34598734958 and ?1234)')
    theorem.hypotheses.extend(['(and type ?1234)'])
    expected = proof_assistant_pb2.Theorem(
        conclusion='(does contain types ?0 and ?1)')
    expected.hypotheses.extend(['(and type ?1)'])
    normalized = normalization_lib.normalize(theorem, consider_hypotheses=True)
    self.assertEqual(normalized.hypotheses[0], expected.hypotheses[0])
    self.assertEqual(normalized.conclusion, expected.conclusion)

  def test_normalize_with_hypotheses_different_types(self):
    theorem = proof_assistant_pb2.Theorem(
        conclusion='(does contain types ?34598734958 and ?1234)')
    theorem.hypotheses.extend(['(and type ?122143)'])
    expected = proof_assistant_pb2.Theorem(
        conclusion='(does contain types ?0 and ?1)')
    expected.hypotheses.extend(['(and type ?2)'])
    normalized = normalization_lib.normalize(theorem, consider_hypotheses=True)
    self.assertEqual(normalized.conclusion, expected.conclusion)
    self.assertEqual(normalized.hypotheses[0], expected.hypotheses[0])

  def test_normalize_ignoring_hypotheses(self):
    theorem = proof_assistant_pb2.Theorem(
        conclusion='(does contain types ?34598734958 and ?1234)')
    theorem.hypotheses.extend(['(and type ?122143)'])
    expected = proof_assistant_pb2.Theorem(
        conclusion='(does contain types ?0 and ?1)')
    normalized = normalization_lib.normalize(theorem, consider_hypotheses=False)
    self.assertEqual(normalized.conclusion, expected.conclusion)
    self.assertEqual(normalized.hypotheses, expected.hypotheses)

  def test_single_genpvar(self):
    theorem = proof_assistant_pb2.Theorem(
        conclusion='(l (v type GEN%PVAR%123) (v type GEN%PVAR%123))')
    expected = proof_assistant_pb2.Theorem(
        conclusion='(l (v type GEN%PVAR%0) (v type GEN%PVAR%0))')
    self.assertEqual(
        normalization_lib.normalize_genpvars(theorem.conclusion),
        expected.conclusion)
    self.assertEqual(normalization_lib.normalize(theorem), expected)

  def test_two_genpvars(self):
    data = ('(f (l (v type GEN%PVAR%123) (v type GEN%PVAR%123)) (l (v type '
            'GEN%PVAR%123) (v type GEN%PVAR%123)))')
    expected = ('(f (l (v type GEN%PVAR%0) (v type GEN%PVAR%0)) (l (v type '
                'GEN%PVAR%0) (v type GEN%PVAR%0)))')
    self.assertEqual(normalization_lib.normalize_genpvars(data), expected)

  def test_two_genpvars2(self):
    data = ('(f (l (v type GEN%PVAR%123) (v type GEN%PVAR%123)) (l (v type '
            'GEN%PVAR%999) (v type GEN%PVAR%999)))')
    expected = ('(f (l (v type GEN%PVAR%0) (v type GEN%PVAR%0)) (l (v type '
                'GEN%PVAR%0) (v type GEN%PVAR%0)))')
    self.assertEqual(normalization_lib.normalize_genpvars(data), expected)

  def test_free_genpvars(self):
    data = '(f (l (v type GEN%PVAR%123) otherexpr) (v type GEN%PVAR%9))'
    expected = '(f (l (v type GEN%PVAR%0) otherexpr) (v type GEN%PVAR%9))'
    self.assertEqual(normalization_lib.normalize_genpvars(data), expected)

  def test_free_genpvars_2(self):
    data = ('(f (l (v type GEN%PVAR%123) (v type GEN%PVAR%123)) (v type '
            'GEN%PVAR%9))')
    expected = ('(f (l (v type GEN%PVAR%0) (v type GEN%PVAR%0)) (v type '
                'GEN%PVAR%9))')
    self.assertEqual(normalization_lib.normalize_genpvars(data), expected)

  def test_nested_genpvars(self):
    data = ('(f (l (v type GEN%PVAR%123) (l (v type GEN%PVAR%333) (f (v type '
            'GEN%PVAR%333) (v type GEN%PVAR%123)))) (l (v type GEN%PVAR%999) (v'
            ' type GEN%PVAR%999)))')
    expected = ('(f (l (v type GEN%PVAR%0) (l (v type GEN%PVAR%1) (f (v type '
                'GEN%PVAR%1) (v type GEN%PVAR%0)))) (l (v type GEN%PVAR%0) (v '
                'type GEN%PVAR%0)))')
    self.assertEqual(normalization_lib.normalize_genpvars(data), expected)

  def test_nested_genpvars_twice_normalized(self):
    data = ('(f (l (v type GEN%PVAR%123) (l (v type GEN%PVAR%333) (f (v type '
            'GEN%PVAR%333) (v type GEN%PVAR%123)))) (l (v type GEN%PVAR%999) (v'
            ' type GEN%PVAR%999)))')
    expected = ('(f (l (v type GEN%PVAR%0) (l (v type GEN%PVAR%1) (f (v type '
                'GEN%PVAR%1) (v type GEN%PVAR%0)))) (l (v type GEN%PVAR%0) (v '
                'type GEN%PVAR%0)))')
    self.assertEqual(
        normalization_lib.normalize_genpvars(
            normalization_lib.normalize_genpvars(data)), expected)

  def test_gen_existential_quantifier(self):
    theorem = proof_assistant_pb2.Theorem(
        conclusion='(does contain ?123 but ignores ? quant)')
    expected = proof_assistant_pb2.Theorem(
        conclusion='(does contain ?0 but ignores ? quant)')
    self.assertEqual(normalization_lib.normalize(theorem), expected)

  def test_gen_exists_exactly_one_quantifier(self):
    theorem = proof_assistant_pb2.Theorem(
        conclusion='(does contain ?123 but ignores ?! quant)')
    expected = proof_assistant_pb2.Theorem(
        conclusion='(does contain ?0 but ignores ?! quant)')
    self.assertEqual(normalization_lib.normalize(theorem), expected)

  @parameterized.named_parameters(
      ('ocaml_normalization_is_stable_1',
       '(a (a (c (fun (fun ?2 (bool)) (fun (fun ?2 (bool)) (bool)))'
       ' =) (a (c (fun (fun ?2 (bool)) (fun ?2 (bool))) GSPEC) (l (v ?2 GEN%PV'
       'AR%0) (a (c (fun (fun ?1 (bool)) (bool)) ?) (l (v ?1 x) (a (a (a (c '
       '(fun ?2 (fun (bool) (fun ?2 (bool)))) SETSPEC) (v ?2 GEN%PVAR%0)) (a '
       '(c (fun (fun ?0 (bool)) (bool)) ?) (l (v ?0 y) (a (a (c (fun (bool) '
       '(fun (bool) (bool))) /\\) (a (v (fun ?0 (bool)) P) (v ?0 y))) '
       '(a (a (c (fun ?1 (fun ?1 (bool))) =) (v ?1 x)) '
       '(a (v (fun ?0 ?1) g) (v ?0 y))))))) (a (v (fun ?1 ?2) f) '
       '(v ?1 x)))))))) (a (c (fun (fun ?2 (bool)) (fun ?2 '
       '(bool))) GSPEC) (l (v ?2 GEN%PVAR%0) (a (c (fun (fun ?0 (bool)) (bool)'
       ') ?) (l (v ?0 y) (a (a (a (c (fun ?2 (fun (bool) (fun ?2 (bool)))) SET'
       'SPEC) (v ?2 GEN%PVAR%0)) (a (v (fun ?0 (bool)) P) (v ?0 y))) (a (v (fu'
       'n ?1 ?2) f) (a (v (fun ?0 ?1) g) (v ?0 y)))))))))'),
      ('ocaml_normalization_is_stable_1b',
       '(a (a (c (fun (fun ?2 (bool)) (fun (fun ?2 (bool)) (bool)))'
       ' =) (a (c (fun (fun ?2 (bool)) (fun ?2 (bool))) GSPEC) (l (v ?2 GEN%PV'
       'AR%0) (a (c (fun (fun ?1 (bool)) (bool)) ?) (l (v ?1 x) (a (a (a (c '
       '(fun ?2 (fun (bool) (fun ?2 (bool)))) SETSPEC) (v ?2 GEN%PVAR%0)) (a '
       '(c (fun (fun ?0 (bool)) (bool)) ?) (l (v ?0 y) (a (a (c (fun (bool) '
       '(fun (bool) (bool))) /\\) (a (v (fun ?0 (bool)) P) (v ?0 y))) '
       '(a (a (c (fun ?1 (fun ?1 (bool))) =) (v ?1 x)) '
       '(a (v (fun ?0 ?1) g) (v ?0 y))))))) (a (v (fun ?1 ?2) f) '
       '(v ?1 x)))))))) (a (c (fun (fun ?2 (bool)) (fun ?2 '
       '(bool))) GSPEC) (l (v ?2 GEN%PVAR%0) (a (c (fun (fun ?0 (bool)) (bool)'
       ') ?) (l (v ?0 y) (a (a (a (c (fun ?2 (fun (bool) (fun ?2 (bool)))) SET'
       'SPEC) (v ?2 GEN%PVAR%0)) (a (v (fun ?0 (bool)) P) (v ?0 y))) (a (v (fu'
       'n ?1 ?2) f) (a (v (fun ?0 ?1) g) (v ?0 y)))))))))'),
      ('from_ocaml_1',
       '(a (c (fun (fun (prod ?22533 ?22527) (bool)) (bool)) !) (l (v (prod ?2'
       '2533 ?22527) h) (a (c (fun (fun ?22533 (bool)) (bool)) !) (l (v ?2253'
       '3 a) (a (c (fun (fun (list (prod ?22533 ?22527)) (bool)) (bool)) !) ('
       'l (v (list (prod ?22533 ?22527)) t) (a (a (c (fun ?22527 (fun ?22527 '
       '(bool))) =) (a (a (c (fun ?22533 (fun (list (prod ?22533 ?22527)) ?22'
       '527)) ASSOC) (v ?22533 a)) (a (a (c (fun (prod ?22533 ?22527) (fun (li'
       'st (prod ?22533 ?22527)) (list (prod ?22533 ?22527)))) CONS) (v (prod '
       '?22533 ?22527) h)) (v (list (prod ?22533 ?22527)) t)))) (a (a (a (c ('
       'fun (bool) (fun ?22527 (fun ?22527 ?22527))) COND) (a (a (c (fun ?22533'
       ' (fun ?22533 (bool))) =) (a (c (fun (prod ?22533 ?22527) ?22533) FST) ('
       'v (prod ?22533 ?22527) h))) (v ?22533 a))) (a (c (fun (prod ?22533 ?22'
       '527) ?22527) SND) (v (prod ?22533 ?22527) h))) (a (a (c (fun ?22533 (f'
       'un (list (prod ?22533 ?22527)) ?22527)) ASSOC) (v ?22533 a)) (v (list '
       '(prod ?22533 ?22527)) t))))))))))'),
      ('from_ocaml_1b',
       '(a (c (fun (fun (prod ?1 ?0) (bool)) (bool)) !) (l (v (prod ?1 ?0) h) '
       '(a (c (fun (fun ?1 (bool)) (bool)) !) (l (v ?1 a) (a (c (fun (fun (list'
       ' (prod ?1 ?0)) (bool)) (bool)) !) (l (v (list (prod ?1 ?0)) t) (a (a (c'
       ' (fun ?0 (fun ?0 (bool))) =) (a (a (c (fun ?1 (fun (list (prod ?1 ?0)) '
       '?0)) ASSOC) (v ?1 a)) (a (a (c (fun (prod ?1 ?0) (fun (list (prod ?1 ?0'
       ')) (list (prod ?1 ?0)))) CONS) (v (prod ?1 ?0) h)) (v (list (prod ?1 ?0'
       ')) t)))) (a (a (a (c (fun (bool) (fun ?0 (fun ?0 ?0))) COND) (a (a (c ('
       'fun ?1 (fun ?1 (bool))) =) (a (c (fun (prod ?1 ?0) ?1) FST) (v (prod ?1'
       ' ?0) h))) (v ?1 a))) (a (c (fun (prod ?1 ?0) ?0) SND) (v (prod ?1 ?0) h'
       '))) (a (a (c (fun ?1 (fun (list (prod ?1 ?0)) ?0)) ASSOC) (v ?1 a)) (v '
       '(list (prod ?1 ?0)) t))))))))))'),
      ('from_ocaml_2',
       '(h () (a (c (fun (fun (fun A B) (bool)) (bool)) !) (l (v (fun A B) f) ('
       'a (a (c (fun (bool) (fun (bool) (bool))) =) (a (c (fun (fun (fun B (boo'
       'l)) (bool)) (bool)) !) (l (v (fun B (bool)) t) (a (c (fun (fun (fun B ('
       'bool)) (bool)) (bool)) !) (l (v (fun B (bool)) t\') (a (a (c (fun (bool'
       ') (fun (bool) (bool))) ==>) (a (a (c (fun (fun A (bool)) (fun (fun A (b'
       'ool)) (bool))) =) (a (c (fun (fun A (bool)) (fun A (bool))) GSPEC) (l '
       '(v A GEN%PVAR%159) (a (c (fun (fun A (bool)) (bool)) ?) (l (v A x) (a '
       '(a (a (c (fun A (fun (bool) (fun A (bool)))) SETSPEC) (v A GEN%PVAR%15'
       '9)) (a (a (c (fun B (fun (fun B (bool)) (bool))) IN) (a (v (fun A B) '
       'f) (v A x))) (v (fun B (bool)) t))) (v A x))))))) (a (c (fun (fun A (b'
       'ool)) (fun A (bool))) GSPEC) (l (v A GEN%PVAR%160) (a (c (fu'
       'n (fun A (bool)) (bool)) ?) (l (v A x) (a (a (a (c (fun A (fun (bool) '
       '(fun A (bool)))) SETSPEC) (v A GEN%PVAR%160)) (a (a (c (fun B (fun (fu'
       'n B (bool)) (bool))) IN) (a (v (fun A B) f) (v A x))) (v (fun B (bool)'
       ') t\'))) (v A x)))))))) (a (a (c (fun (fun B (bool)) (fun (fun B (bool'
       ')) (bool))) =) (v (fun B (bool)) t)) (v (fun B (bool)) t\')))))))) (a '
       '(a (c (fun (fun B (bool)) (fun (fun B (bool)) (bool))) =) (a (a (c (fu'
       'n (fun A B) (fun (fun A (bool)) (fun B (bool)))) IMAGE) (v (fun A B) f'
       ')) (c (fun A (bool)) UNIV))) (c (fun B (bool)) UNIV))))))'),
      ('from_ocaml_2b',
       '(h () (a (c (fun (fun (fun A B) (bool)) (bool)) !) (l (v (fun A B) f) '
       '(a (a (c (fun (bool) (fun (bool) (bool))) =) (a (c (fun (fun (fun B (b'
       'ool)) (bool)) (bool)) !) (l (v (fun B (bool)) t) (a (c (fun (fun (fun '
       'B (bool)) (bool)) (bool)) !) (l (v (fun B (bool)) t\') (a (a (c (fun ('
       'bool) (fun (bool) (bool))) ==>) (a (a (c (fun (fun A (bool)) (fun (fun'
       ' A (bool)) (bool))) =) (a (c (fun (fun A (bool)) (fun A (bool))) GSPEC'
       ') (l (v A GEN%PVAR%0) (a (c (fun (fun A (bool)) (bool)) ?) (l (v A x) '
       '(a (a (a (c (fun A (fun (bool) (fun A (bool)))) SETSPEC) (v A GEN%PVAR'
       '%0)) (a (a (c (fun B (fun (fun B (bool)) (bool))) IN) (a (v (fun A B) '
       'f) (v A x))) (v (fun B (bool)) t))) (v A x))))))) (a (c (fun (fun A (bo'
       'ol)) (fun A (bool))) GSPEC) (l (v A GEN%PVAR%0) (a (c (fun (fun A (bool'
       ')) (bool)) ?) (l (v A x) (a (a (a (c (fun A (fun (bool) (fun A (bool)))'
       ') SETSPEC) (v A GEN%PVAR%0)) (a (a (c (fun B (fun (fun B (bool)) (bool'
       '))) IN) (a (v (fun A B) f) (v A x))) (v (fun B (bool)) t\'))) (v A x))'
       ')))))) (a (a (c (fun (fun B (bool)) (fun (fun B (bool)) (bool))) =) (v'
       ' (fun B (bool)) t)) (v (fun B (bool)) t\')))))))) (a (a (c (fun (fun B'
       ' (bool)) (fun (fun B (bool)) (bool))) =) (a (a (c (fun (fun A B) (fun '
       '(fun A (bool)) (fun B (bool)))) IMAGE) (v (fun A B) f)) (c (fun A (boo'
       'l)) UNIV))) (c (fun B (bool)) UNIV))))))'),
      ('from_ocaml_3',
       '(h () (a (c (fun (fun (fun ?78897 (fun ?78897 ?78897)) (bool)) (bool))'
       ' !) (l (v (fun ?78897 (fun ?78897 ?78897)) op) (a (c (fun (fun (fun ?7'
       '8925 (bool)) (bool)) (bool)) !) (l (v (fun ?78925 (bool)) s) (a (c (fu'
       'n (fun (fun ?78925 ?78897) (bool)) (bool)) !) (l (v (fun ?78925 ?78897'
       ') f) (a (c (fun (fun ?78925 (bool)) (bool)) !) (l (v ?78925 a) (a (a ('
       'c (fun (fun ?78925 (bool)) (fun (fun ?78925 (bool)) (bool))) =) (a (a '
       '(a (c (fun (fun ?78897 (fun ?78897 ?78897)) (fun (fun ?78925 ?78897) ('
       'fun (fun ?78925 (bool)) (fun ?78925 (bool))))) support) (v (fun ?78897'
       ' (fun ?78897 ?78897)) op)) (l (v ?78925 x) (a (a (a (c (fun (bool) (fu'
       'n ?78897 (fun ?78897 ?78897))) COND) (a (a (c (fun ?78925 (fun ?78925'
       ' (bool))) =) (v ?78925 x)) (v ?78925 a))) (a (v (fun ?78925 ?78897) f)'
       ' (v ?78925 x))) (a (c (fun (fun ?78897 (fun ?78897 ?78897)) ?78897) ne'
       'utral) (v (fun ?78897 (fun ?78897 ?78897)) op))))) (v (fun ?78925 (boo'
       'l)) s))) (a (a (a (c (fun (bool) (fun (fun ?78925 (bool)) (fun (fun ?7'
       '8925 (bool)) (fun ?78925 (bool))))) COND) (a (a (c (fun ?78925 (fun (f'
       'un ?78925 (bool)) (bool))) IN) (v ?78925 a)) (v (fun ?78925 (bool)) s)'
       ')) (a (a (a (c (fun (fun ?78897 (fun ?78897 ?78897)) (fun (fun ?78925 '
       '?78897) (fun (fun ?78925 (bool)) (fun ?78925 (bool))))) support) (v (f'
       'un ?78897 (fun ?78897 ?78897)) op)) (v (fun ?78925 ?78897) f)) (a (a ('
       'c (fun ?78925 (fun (fun ?78925 (bool)) (fun ?78925 (bool)))) INSERT) ('
       'v ?78925 a)) (c (fun ?78925 (bool)) EMPTY)))) (c (fun ?78925 (bool)) E'
       'MPTY))))))))))))'),
      ('from_ocaml_3b',
       '(h () (a (c (fun (fun (fun ?1 (fun ?1 ?1)) (bool)) (bool)) !) (l (v ('
       'fun ?1 (fun ?1 ?1)) op) (a (c (fun (fun (fun ?0 (bool)) (bool)) (bool'
       ')) !) (l (v (fun ?0 (bool)) s) (a (c (fun (fun (fun ?0 ?1) (bool)) (b'
       'ool)) !) (l (v (fun ?0 ?1) f) (a (c (fun (fun ?0 (bool)) (bool)) !) ('
       'l (v ?0 a) (a (a (c (fun (fun ?0 (bool)) (fun (fun ?0 (bool)) (bool))'
       ') =) (a (a (a (c (fun (fun ?1 (fun ?1 ?1)) (fun (fun ?0 ?1) (fun (fun'
       ' ?0 (bool)) (fun ?0 (bool))))) support) (v (fun ?1 (fun ?1 ?1)) op)) '
       '(l (v ?0 x) (a (a (a (c (fun (bool) (fun ?1 (fun ?1 ?1))) COND) (a (a'
       ' (c (fun ?0 (fun ?0 (bool))) =) (v ?0 x)) (v ?0 a))) (a (v (fun ?0 ?1'
       ') f) (v ?0 x))) (a (c (fun (fun ?1 (fun ?1 ?1)) ?1) neutral) (v (fun '
       '?1 (fun ?1 ?1)) op))))) (v (fun ?0 (bool)) s))) (a (a (a (c (fun (boo'
       'l) (fun (fun ?0 (bool)) (fun (fun ?0 (bool)) (fun ?0 (bool))))) COND)'
       ' (a (a (c (fun ?0 (fun (fun ?0 (bool)) (bool))) IN) (v ?0 a)) (v (fun'
       ' ?0 (bool)) s))) (a (a (a (c (fun (fun ?1 (fun ?1 ?1)) (fun (fun ?0 ?'
       '1) (fun (fun ?0 (bool)) (fun ?0 (bool))))) support) (v (fun ?1 (fun ?'
       '1 ?1)) op)) (v (fun ?0 ?1) f)) (a (a (c (fun ?0 (fun (fun ?0 (bool)) '
       '(fun ?0 (bool)))) INSERT) (v ?0 a)) (c (fun ?0 (bool)) EMPTY)))) (c ('
       'fun ?0 (bool)) EMPTY))))))))))))'),
      ('from_ocaml_4',
       '(h () (a (a (c (fun (fun (fun ?125588 ?125588) (bool)) (fun (fun (fun '
       '?125588 ?125588) (bool)) (bool))) =) (a (c (fun (fun (fun ?125588 ?125'
       '588) (bool)) (fun (fun ?125588 ?125588) (bool))) GSPEC) (l (v (fun ?12'
       '5588 ?125588) GEN%PVAR%403) (a (c (fun (fun (fun ?125588 ?125588) (boo'
       'l)) (bool)) ?) (l (v (fun ?125588 ?125588) p) (a (a (a (c (fun (fun ?1'
       '25588 ?125588) (fun (bool) (fun (fun ?125588 ?125588) (bool)))) SETSPE'
       'C) (v (fun ?125588 ?125588) GEN%PVAR%403)) (a (a (c (fun (bool) (fun ('
       'bool) (bool))) /\\) (a (a (c (fun (fun ?125588 ?125588) (fun (fun ?1255'
       '88 (bool)) (bool))) permutes) (v (fun ?125588 ?125588) p)) (v (fun ?12'
       '5588 (bool)) s))) (a (v (fun (fun ?125588 ?125588) (bool)) Q) (v (fun '
       '?125588 ?125588) p)))) (v (fun ?125588 ?125588) p))))))) (a (c (fun (f'
       'un (fun ?125588 ?125588) (bool)) (fun (fun ?125588 ?125588) (bool))) G'
       'SPEC) (l (v (fun ?125588 ?125588) GEN%PVAR%405) (a (c (fun (fun (fun ?'
       '125588 ?125588) (bool)) (bool)) ?) (l (v (fun ?125588 ?125588) p) (a ('
       'a (a (c (fun (fun ?125588 ?125588) (fun (bool) (fun (fun ?125588 ?1255'
       '88) (bool)))) SETSPEC) (v (fun ?125588 ?125588) GEN%PVAR%405)) (a (a ('
       'c (fun (bool) (fun (bool) (bool))) /\\) (a (a (c (fun (fun ?125588 ?125'
       '588) (fun (fun (fun ?125588 ?125588) (bool)) (bool))) IN) (v (fun ?125'
       '588 ?125588) p)) (a (c (fun (fun (fun ?125588 ?125588) (bool)) (fun (f'
       'un ?125588 ?125588) (bool))) GSPEC) (l (v (fun ?125588 ?125588) GEN%PV'
       'AR%404) (a (c (fun (fun (fun ?125588 ?125588) (bool)) (bool)) ?) (l (v'
       ' (fun ?125588 ?125588) p) (a (a (a (c (fun (fun ?125588 ?125588) (fun '
       '(bool) (fun (fun ?125588 ?125588) (bool)))) SETSPEC) (v (fun ?125588 ?'
       '125588) GEN%PVAR%404)) (a (a (c (fun (fun ?125588 ?125588) (fun (fun ?'
       '125588 (bool)) (bool))) permutes) (v (fun ?125588 ?125588) p)) (v (fun'
       ' ?125588 (bool)) s))) (v (fun ?125588 ?125588) p)))))))) (a (v (fun (f'
       'un ?125588 ?125588) (bool)) Q) (v (fun ?125588 ?125588) p)))) (v (fun '
       '?125588 ?125588) p))))))))'),
      ('from_ocaml_4b',
       '(h () (a (a (c (fun (fun (fun ?0 ?0) (bool)) (fun (fun (fun ?0 ?0) (bo'
       'ol)) (bool))) =) (a (c (fun (fun (fun ?0 ?0) (bool)) (fun (fun ?0 ?0) '
       '(bool))) GSPEC) (l (v (fun ?0 ?0) GEN%PVAR%0) (a (c (fun (fun (fun ?0 '
       '?0) (bool)) (bool)) ?) (l (v (fun ?0 ?0) p) (a (a (a (c (fun (fun ?0 ?'
       '0) (fun (bool) (fun (fun ?0 ?0) (bool)))) SETSPEC) (v (fun ?0 ?0) GEN%'
       'PVAR%0)) (a (a (c (fun (bool) (fun (bool) (bool))) /\\) (a (a (c (fun '
       '(fun ?0 ?0) (fun (fun ?0 (bool)) (bool))) permutes) (v (fun ?0 ?0) p)'
       ') (v (fun ?0 (bool)) s))) (a (v (fun (fun ?0 ?0) (bool)) Q) (v (fun ?0'
       ' ?0) p)))) (v (fun ?0 ?0) p))))))) (a (c (fun (fun (fun ?0 ?0) (bool))'
       ' (fun (fun ?0 ?0) (bool))) GSPEC) (l (v (fun ?0 ?0) GEN%PVAR%0) (a (c '
       '(fun (fun (fun ?0 ?0) (bool)) (bool)) ?) (l (v (fun ?0 ?0) p) (a (a (a'
       ' (c (fun (fun ?0 ?0) (fun (bool) (fun (fun ?0 ?0) (bool)))) SETSPEC) ('
       'v (fun ?0 ?0) GEN%PVAR%0)) (a (a (c (fun (bool) (fun (bool) (bool))) /'
       '\\) (a (a (c (fun (fun ?0 ?0) (fun (fun (fun ?0 ?0) (bool)) (bool))) IN'
       ') (v (fun ?0 ?0) p)) (a (c (fun (fun (fun ?0 ?0) (bool)) (fun (fun ?0 '
       '?0) (bool))) GSPEC) (l (v (fun ?0 ?0) GEN%PVAR%1) (a (c (fun (fun (fun'
       ' ?0 ?0) (bool)) (bool)) ?) (l (v (fun ?0 ?0) p) (a (a (a (c (fun (fun '
       '?0 ?0) (fun (bool) (fun (fun ?0 ?0) (bool)))) SETSPEC) (v (fun ?0 ?0) '
       'GEN%PVAR%1)) (a (a (c (fun (fun ?0 ?0) (fun (fun ?0 (bool)) (bool))) p'
       'ermutes) (v (fun ?0 ?0) p)) (v (fun ?0 (bool)) s))) (v (fun ?0 ?0) p))'
       ')))))) (a (v (fun (fun ?0 ?0) (bool)) Q) (v (fun ?0 ?0) p)))) (v (fun '
       '?0 ?0) p))))))))'),
      ('from_ocaml_5',
       '(h () (a (a (c (fun (fun ?271134 (bool)) (fun (fun ?271134 (bool)) (bo'
       'ol))) =) (a (c (fun (fun ?271134 (bool)) (fun ?271134 (bool))) GSPEC) '
       '(l (v ?271134 GEN%PVAR%942) (a (c (fun (fun ?271134 (bool)) (bool)) ?)'
       ' (l (v ?271134 y) (a (a (a (c (fun ?271134 (fun (bool) (fun ?271134 (b'
       'ool)))) SETSPEC) (v ?271134 GEN%PVAR%942)) (a (a (c (fun (bool) (fun ('
       'bool) (bool))) /\\) (a (a (c (fun ?271134 (fun (fun ?271134 (bool)) (bo'
       'ol))) IN) (v ?271134 y)) (a (c (fun (fun ?271134 (bool)) (fun ?271134 '
       '(bool))) GSPEC) (l (v ?271134 GEN%PVAR%941) (a (c (fun (fun ?271135 (b'
       'ool)) (bool)) ?) (l (v ?271135 x) (a (a (a (c (fun ?271134 (fun (bool)'
       ' (fun ?271134 (bool)))) SETSPEC) (v ?271134 GEN%PVAR%941)) (a (a (c (f'
       'un ?271135 (fun (fun ?271135 (bool)) (bool))) IN) (v ?271135 x)) (v (f'
       'un ?271135 (bool)) s))) (a (v (fun ?271135 ?271134) f) (v ?271135 x)))'
       ')))))) (a (v (fun ?271134 (bool)) P) (v ?271134 y)))) (v ?271134 y))))'
       '))) (a (a (c (fun (fun ?271135 ?271134) (fun (fun ?271135 (bool)) (fun'
       ' ?271134 (bool)))) IMAGE) (v (fun ?271135 ?271134) f)) (a (c (fun (fun'
       ' ?271135 (bool)) (fun ?271135 (bool))) GSPEC) (l (v ?271135 GEN%PVAR%9'
       '43) (a (c (fun (fun ?271135 (bool)) (bool)) ?) (l (v ?271135 x) (a (a '
       '(a (c (fun ?271135 (fun (bool) (fun ?271135 (bool)))) SETSPEC) (v ?271'
       '135 GEN%PVAR%943)) (a (a (c (fun (bool) (fun (bool) (bool))) /\\) (a (a'
       ' (c (fun ?271135 (fun (fun ?271135 (bool)) (bool))) IN) (v ?271135 x))'
       ' (v (fun ?271135 (bool)) s))) (a (v (fun ?271134 (bool)) P) (a (v (fun'
       ' ?271135 ?271134) f) (v ?271135 x))))) (v ?271135 x)))))))))'),
      ('from_ocaml_5b',
       '(h () (a (a (c (fun (fun ?1 (bool)) (fun (fun ?1 (bool)) (bool))) =) (a'
       ' (c (fun (fun ?1 (bool)) (fun ?1 (bool))) GSPEC) (l (v ?1 GEN%PVAR%0) '
       '(a (c (fun (fun ?1 (bool)) (bool)) ?) (l (v ?1 y) (a (a (a (c (fun ?1 '
       '(fun (bool) (fun ?1 (bool)))) SETSPEC) (v ?1 GEN%PVAR%0)) (a (a (c (fu'
       'n (bool) (fun (bool) (bool))) /\\) (a (a (c (fun ?1 (fun (fun ?1 (bool)'
       ') (bool))) IN) (v ?1 y)) (a (c (fun (fun ?1 (bool)) (fun ?1 (bool))) '
       'GSPEC) (l (v ?1 GEN%PVAR%1) (a (c (fun (fun ?0 (bool)) (bool)) ?) (l '
       '(v ?0 x) (a (a (a (c (fun ?1 (fun (bool) (fun ?1 (bool)))) SETSPEC) ('
       'v ?1 GEN%PVAR%1)) (a (a (c (fun ?0 (fun (fun ?0 (bool)) (bool))) IN) '
       '(v ?0 x)) (v (fun ?0 (bool)) s))) (a (v (fun ?0 ?1) f) (v ?0 x)))))))'
       ')) (a (v (fun ?1 (bool)) P) (v ?1 y)))) (v ?1 y))))))) (a (a (c (fun '
       '(fun ?0 ?1) (fun (fun ?0 (bool)) (fun ?1 (bool)))) IMAGE) (v (fun ?0 '
       '?1) f)) (a (c (fun (fun ?0 (bool)) (fun ?0 (bool))) GSPEC) (l (v ?0 G'
       'EN%PVAR%0) (a (c (fun (fun ?0 (bool)) (bool)) ?) (l (v ?0 x) (a (a (a'
       ' (c (fun ?0 (fun (bool) (fun ?0 (bool)))) SETSPEC) (v ?0 GEN%PVAR%0))'
       ' (a (a (c (fun (bool) (fun (bool) (bool))) /\\) (a (a (c (fun ?0 (fun '
       '(fun ?0 (bool)) (bool))) IN) (v ?0 x)) (v (fun ?0 (bool)) s))) (a (v '
       '(fun ?1 (bool)) P) (a (v (fun ?0 ?1) f) (v ?0 x))))) (v ?0 x)))))))))'))
  def test_idempotency(self, expr):
    # Tests if python normalization is idempotent.
    theorem = proof_assistant_pb2.Theorem(conclusion=expr)
    normalized = proof_assistant_pb2.Theorem()
    normalized.CopyFrom(normalization_lib.normalize(theorem))
    self.assertEqual(normalization_lib.normalize(normalized), normalized)


if __name__ == '__main__':
  tf.test.main()
