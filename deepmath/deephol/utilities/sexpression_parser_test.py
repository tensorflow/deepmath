"""Tests for deephol.utilities.sexpression_parser."""
from absl.testing import parameterized
import tensorflow.compat.v1 as tf

from deepmath.deephol.utilities import sexpression_parser


class SexpressionParserTest(parameterized.TestCase):

  def test_empty_string(self):
    self.assertTrue(sexpression_parser.is_bare_word(''))

  @parameterized.named_parameters(
      ('simple_bare_word', 'asdf', True),
      ('bare_word_with_newline', 'as\ndf', True),
      ('bare_word_with_backslash', 'as\\df', True),
      ('two_words', 'as df', False), ('dont_contain_parens1', 'as)df', False),
      ('dont_contain_parens2', 'as(df', False),
      ('dont_contain_parens3', 'as()df', False),
      ('dont_contain_parens4', 'as)(df', False),
      ('dont_contain_parens5', ')asdf', False),
      ('dont_contain_parens6', 'asdf(', False),
      ('dont_contain_parens7', '(asdf', False),
      ('dont_contain_parens8', 'asdf)', False),
      ('bare_words_dont_start_with_space', ' asdf', False),
      ('bare_words_dont_end_with_space', 'adsf ', False))
  def test_bare_word(self, word, is_bare_word):
    self.assertEqual(sexpression_parser.is_bare_word(word), is_bare_word)

  def test_single_child(self):
    sexp = '(asdf)'
    root = sexpression_parser.to_tree(sexp)
    self.assertLen(root.children, 1)

  def test_two_children(self):
    sexp = '(asdf asdf)'
    root = sexpression_parser.to_tree(sexp)
    self.assertLen(root.children, 2)

  def test_docstring_example(self):
    sexp = '(word1 word1 (word1) () (() ()))'
    root = sexpression_parser.to_tree(sexp)
    self.assertLen(root.children, 5)
    self.assertEqual([repr(child) for child in root.children],
                     ['word1', 'word1', '(word1)', '()', '(() ())'])

  @parameterized.named_parameters(
      ('two_sexps_exn', '(asdf) (asdf)'), ('paren_exn', '('),
      ('parens_exn', '(('), ('closing_paren_exn', ')'), ('space_exn', ' '),
      ('closing_parens_exn', '))'), ('partially_matched_parens_exn', '(()'),
      ('anti_matching_parens_exn', ')('),
      ('almost_sexp_exn', '(asdf (asdf)) (())'))
  def test_invalid_sexp(self, string):
    with self.assertRaises(sexpression_parser.SExpParseError):
      sexpression_parser.to_tree(string)

  def test_grand_children(self):
    sexp = '(asdf1 (asdf2 asdf3))'
    root = sexpression_parser.to_tree(sexp)
    self.assertLen(root.children, 2)
    self.assertLen(root.children[1].children, 2)


if __name__ == '__main__':
  tf.test.main()
