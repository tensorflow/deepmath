# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Functions for working with CNF formulas in JSON format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_bool('cnf_allow_toplevel_func', True,
                  'Allow functions to appear at the top level of CNF formulas.')


def raise_error_with_keys(name, dic):
  raise ValueError('Invalid %s with keys: %s' %
                   (name, ', '.join(str(k) for k in dic.keys())))


def validate_cnf(cnf):
  """Check that a CNF formula in JSON representation is valid."""
  if 'clauses' not in cnf:
    raise ValueError('Invalid cnf: lacks "clauses"')

  for literal in cnf['clauses']:
    validate_literal(literal)


def validate_literal(literal):
  """Check that a CNF literal in JSON representation is valid."""
  if 'positive' not in literal:
    raise ValueError('Invalid literal: lacks "positive"')

  if not FLAGS.cnf_allow_toplevel_func:
    if sum(
        ['equal' in literal, 'pred' in literal and 'params' in literal]) != 1:
      raise_error_with_keys('literal', literal)

  if 'equal' in literal:
    params = literal['equal']
  elif 'func' in literal or 'pred' in literal:
    params = literal['params']
  elif 'pred' in literal:
    params = literal['params']

  for param in params:
    validate_term(param)


def validate_term(term):
  """Check that a CNF term in JSON representation is valid."""
  if (sum(['func' in term and 'params' in term, 'number' in term, 'var' in term
          ]) != 1):
    raise_error_with_keys('term', term)

  if 'func' in term:
    for param in term['params']:
      validate_term(param)


def unparse_cnf(cnf):
  """Convert a CNF formula in a JSON-format dictionary into TPTP syntax."""
  return ' | '.join(unparse_literal(literal) for literal in cnf['clauses'])


def unparse_literal(literal):
  """Convert a CNF literal in a JSON-format dictionary into TPTP syntax."""
  if 'equal' in literal:
    terms = literal['equal']
    assert len(terms) == 2
    return ' '.join([unparse_term(terms[0]), '=' if 'positive' not in literal or
                     literal['positive'] else '!=', unparse_term(terms[1])])

  unparsed_func = unparse_function(literal)
  if literal['positive']:
    return unparsed_func
  else:
    return '~' + unparsed_func


def unparse_function(function):
  """Convert a CNF function in a JSON-format dictionary into TPTP syntax."""
  if 'error' in function:
    return 'ERROR(' + function['error'] + ')'

  func_name = function['func'] if 'func' in function else function['pred']
  params = function['params']

  result = [func_name]
  if params:
    result += ['(', ', '.join(unparse_term(param) for param in params), ')']
  return ''.join(result)


def unparse_term(term):
  """Convert a CNF term in a JSON-format dictionary into TPTP syntax."""
  if 'error' in term:
    return 'ERROR(' + term['error'] + ')'

  if 'func' in term or 'pred' in term:
    return unparse_function(term)
  elif 'equal' in term:
    # This is illegal, but the generative model might produce it.
    return unparse_literal(term)
  elif 'number' in term:
    return term['number']
  elif 'var' in term:
    return term['var']

  raise ValueError('Invalid term: ' + repr(term))
