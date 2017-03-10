# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for parser.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from deepmath.holstep import parser


class ParserTest(tf.test.TestCase):

  def _check(self, term, correct):
    t = parser.parse_term(term)
    s = parser.show_sexp(t)
    self.assertEqual(s, correct)

  def testParse(self):
    self._check('f x', '(f x)')
    self._check('(f x) y', '(f x y)')
    self._check('f (x y)', '(f (x y))')

    self._check('x + y', '(+ x y)')
    self._check('(x + y) * z', '(* (+ x y) z)')
    self._check('x + (y * z)', '(+ x (* y z))')

    self._check('|- x', '(|- x)')
    self._check('a |- x', '(|- a x)')
    self._check('a, b |- x', '(|- a b x)')
    self._check('a , b |- x', '(|- a b x)')
    self._check(r'\x. (x + y)', r'(\ x (+ x y))')
    self._check('!x. (x = 0)', '(! x (= x 0))')
    self._check('?x. (x = 0)', '(? x (= x 0))')
    self._check('?!x. (x = 0)', '(?! x (= x 0))')
    self._check(r'(\x. x) y', r'((\ x x) y)')
    self._check('(lambdax. x) y', r'((\ x x) y)')
    self._check('(@x. x) y', '((@ x x) y)')
    self._check(r'(\GEN%PVAR%2364. GEN%PVAR%2364) y',
                r'((\ GEN%PVAR%2364 GEN%PVAR%2364) y)')
    self._check(r'(\x. x) ((f y) x)', r'((\ x x) (f y x))')

  def testInvalid(self):
    for s in '(a)', 'a b c d', 'a b c d e', 'a (b c d e)':
      with self.assertRaisesRegexp(ValueError, 'Invalid intermediate parse'):
        parser.parse_term(s)


if __name__ == '__main__':
  tf.test.main()
