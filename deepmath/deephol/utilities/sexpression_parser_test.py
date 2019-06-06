"""Tests for deephol.utilities.sexpression_parser."""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf

from deepmath.deephol.utilities import sexpression_parser


class SexpressionParserTest(parameterized.TestCase):

  def test_empty_string(self):
    self.assertTrue(sexpression_parser.is_start_of_word('', 0))
    self.assertTrue(sexpression_parser.is_bare_word(''))
    with self.assertRaises(Exception):
      sexpression_parser.is_start_of_word('', 1)
    with self.assertRaises(Exception):
      sexpression_parser.is_start_of_word('', 2)
    with self.assertRaises(Exception):
      sexpression_parser.is_start_of_word('', -1)
    self.assertTrue(sexpression_parser.is_end_of_word('', 0))
    with self.assertRaises(Exception):
      sexpression_parser.is_end_of_word('', 1)
    with self.assertRaises(Exception):
      sexpression_parser.is_end_of_word('', 2)
    with self.assertRaises(Exception):
      sexpression_parser.is_end_of_word('', -1)

  @parameterized.named_parameters(
      ('single_word_1', 'trolololo'), ('single_word_2', 'asdf'),
      ('single_word_with_specials', 'trololo!@#$%^&*_-"\';:+={}[]|.,?~`\\lo'),
      ('single_word_malformed_escapes', 'trololo!@#$%^&*_-"\';:+={}[]|.,?~`\\/'
       '\\//\\\/\nlo'))  # pylint: disable=anomalous-backslash-in-string
  def test_single_word(self, word):
    self.assertTrue(sexpression_parser.is_start_of_word(word, 0))
    for idx in range(1, len(word)):
      self.assertFalse(sexpression_parser.is_start_of_word(word, idx))
    self.assertTrue(sexpression_parser.is_end_of_word(word, len(word)))
    with self.assertRaises(Exception):
      sexpression_parser.is_end_of_word(word, len(word) + 1)
    with self.assertRaises(Exception):
      sexpression_parser.is_start_of_word(word, -1)

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

  def test_two_words(self):
    word = 'trolo !@#$%^&*_-"\';:+={}[]|.,?~`\\lo'
    self.assertTrue(sexpression_parser.is_start_of_word(word, 0))
    self.assertTrue(sexpression_parser.is_end_of_word(word, 5))
    self.assertTrue(sexpression_parser.is_start_of_word(word, 6))
    for idx in range(1, len(word)):
      if idx == 6:
        continue
      self.assertFalse(sexpression_parser.is_start_of_word(word, idx))
    self.assertTrue(sexpression_parser.is_end_of_word(word, len(word)))
    self.assertEqual(sexpression_parser.end_of_word(word, 0), 5)
    with self.assertRaises(Exception):
      sexpression_parser.end_of_word(word, 1)
    with self.assertRaises(Exception):
      sexpression_parser.children(word)

  def test_single_child(self):
    sexp = '(asdf)'
    self.assertLen(sexpression_parser.children(sexp), 1)

  def test_two_children(self):
    sexp = '(asdf asdf)'
    self.assertLen(sexpression_parser.children(sexp), 2)

  def test_docstring_example(self):
    sexp = '(word1 word1 (word1) () (() ()))'
    children = sexpression_parser.children(sexp)
    self.assertLen(children, 5)
    self.assertEqual(children, ['word1', 'word1', '(word1)', '()', '(() ())'])

  @parameterized.named_parameters(
      ('two_sexps_exn', '(asdf) (asdf)'), ('paren_exn', '('),
      ('parens_exn', '(('), ('closing_paren_exn', ')'), ('space_exn', ' '),
      ('closing_parens_exn', '))'), ('partially_matched_parens_exn', '(()'),
      ('anti_matching_parens_exn', ')('),
      ('almost_sexp_exn', '(asdf (asdf)) (())'))
  def test_invalid_sexp(self, string):
    with self.assertRaises(Exception):
      sexpression_parser.children(string)

  def test_grand_children(self):
    sexp = '(asdf1 (asdf2 asdf3))'
    self.assertLen(sexpression_parser.children(sexp), 2)
    self.assertLen(
        sexpression_parser.children(sexpression_parser.children(sexp)[1]), 2)


if __name__ == '__main__':
  tf.test.main()
