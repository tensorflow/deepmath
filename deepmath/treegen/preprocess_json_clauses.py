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
"""Collect metadata from a dataset of JSON clauses.

Outputs the following:
  - All function names which occur in the data.
  - Arities of all functions.
  - All variable names.
  - All numbers seen in the data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input', None, 'JSON-lines files containing all clauses.')
flags.DEFINE_string('output', None, 'Directory to write output.')


class FunctionsAndVariablesCounter(object):
  """Accumulate functions and variables in the data."""

  def __init__(self):
    self.func_names = collections.Counter()
    self.func_arities = collections.defaultdict(set)
    self.var_names = collections.Counter()
    self.numbers = collections.Counter()
    self.func_is_predicate = collections.defaultdict(set)

  def add_file(self, filename):
    with tf.gfile.GFile(filename) as f:
      for line in f:
        clause = json.loads(line)
        self.accumulate(clause)

  def accumulate(self, fragment):
    """Add parts of this fragment to the collection."""
    if 'number' in fragment:
      self.numbers[int(fragment['number'])] += 1
      return

    if 'clauses' in fragment:
      for clause in fragment['clauses']:
        self.accumulate(clause)
      return

    if 'var' in fragment:
      self.var_names[fragment['var']] += 1
      return

    if 'pred' in fragment:
      name = fragment['pred']
      params = fragment['params']
      predicate = True
    elif 'func' in fragment:
      name = fragment['func']
      params = fragment['params']
      predicate = False
    elif 'equal' in fragment:
      name = '__equal__'
      params = fragment['equal']
      predicate = True
    else:
      raise ValueError(fragment.keys())

    self.func_names[name] += 1
    self.func_arities[name].add(len(params))
    self.func_is_predicate[name].add(predicate)
    for param in params:
      self.accumulate(param)

  def finalize(self):
    """Called after all accumulate() calls."""
    for func_name, arities in self.func_arities.items():
      if len(arities) != 1:
        tf.logging.warn('%s observed with arities %s', func_name, arities)
      self.func_arities[func_name] = arities.pop()

    for func_name, is_predicates in self.func_is_predicate.items():
      if len(is_predicates) != 1:
        tf.logging.warn('%s observed both as predicate and function')
      self.func_is_predicate[func_name] = is_predicates.pop()

    self.func_names = [(n, self.func_is_predicate[n])
                       for n, _ in self.func_names.most_common()]
    self.var_names = [n for n, _ in self.var_names.most_common()]
    self.numbers = [n for n, _ in self.numbers.most_common()]


def main(unused_argv):
  counter = FunctionsAndVariablesCounter()
  for filename in tf.gfile.Glob(FLAGS.input):
    counter.add_file(filename)
  counter.finalize()

  with tf.gfile.GFile(FLAGS.output, 'w') as f:
    ordered_func_arities = collections.OrderedDict()
    for func_name, _ in counter.func_names:
      ordered_func_arities[func_name] = counter.func_arities[func_name]

    d = collections.OrderedDict([
        ('func_names', counter.func_names),
        ('func_arities', ordered_func_arities),
        ('var_names', counter.var_names), ('numbers', counter.numbers)
    ])
    d['var_names'].sort(key=lambda s: int(s[1:]))
    d['numbers'].sort()
    d['numbers'] = [str(x) for x in d['numbers']]

    json.dump(d, f, indent=0, separators=(',', ': '))


if __name__ == '__main__':
  tf.app.run()
