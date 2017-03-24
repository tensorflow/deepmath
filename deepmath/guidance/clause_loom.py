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
"""Loom model processing ProverClauseExamples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_fold.public import loom
from deepmath.guidance import gen_clause_ops


# TypeShapes
VOCAB_ID = loom.TypeShape(tf.int64, (), 'vocab_id')


def weave_clauses(examples, vocab, embed, conjecture_apply, conjecture_not,
                  conjecture_or, conjecture_and, clause_apply, clause_not,
                  clause_or, combine, shuffle=True, seed=None):
  """Weave serialized ProverClauseExamples using TensorLoom.

  Computes results for a batch of ProverClauseExamples protos.  For each
  ProverClauseExamples with at least one positive and negative, one positive
  and one negative are selected and reduced to results according to the given
  loom ops.

  In the description of the LoomOps below, vocab_id must be VOCAB_ID.

  Args:
    examples: 1-D `string` tensor of serialized `ProverClauseExamples`.
    vocab: Path to vocabulary file.
    embed: LoomOp to embed vocabulary ids: vocab_id -> embedding.
    conjecture_apply: LoomOp for curried function application:
        embedding -> embedding -> embedding.
    conjecture_not: LoomOp for negation: embedding -> embedding.
    conjecture_or: LoomOp for or: embedding -> embedding -> embedding.
    conjecture_and: LooOp for and: embedding -> embedding -> embedding.
    clause_apply: LoomOp for curried function application:
        embedding -> embedding -> embedding.
    clause_not: LoomOp for negation: embedding -> embedding.
    clause_or: LoomOp for or: embedding -> embedding -> embedding.
    combine : LoomOp to merge results:
        conjecture_embedding -> clause_embedding -> combination.
    shuffle: Whether to randomly shuffle ands and ors.
    seed: Optional seed for random number generation.

  Returns:
    combinations: The results of combine for each positive and negative.
    labels: The labels of each positive and negative.
  """
  def weaver_op(**kwds):
    seed1, seed2 = tf.get_seed(seed)
    return gen_clause_ops.random_clauses_weaver(
        examples=examples, vocab=vocab, shuffle=shuffle, seed=seed1,
        seed2=seed2, **kwds)

  ops = {'embed': embed, 'conjecture_apply': conjecture_apply,
         'conjecture_not': conjecture_not, 'conjecture_or': conjecture_or,
         'conjecture_and': conjecture_and, 'clause_apply': clause_apply,
         'clause_not': clause_not, 'clause_or': clause_or,
         'combine': combine}
  label = loom.TypeShape(tf.bool, (), 'label')
  clause_loom = loom.Loom(named_ops=ops, extra_type_shapes=[label],
                          weaver_op=weaver_op)
  combination, = combine.output_type_shapes
  return (clause_loom.output_tensor(combination),
          clause_loom.output_tensor(label))


def weave_fast_clauses(clauses,
                       embed,
                       apply_,
                       not_,
                       or_,
                       and_=None,
                       shuffle=True,
                       seed=None):
  """Weave serialized FastClauses using TensorLoom.

  Computes embeddings for a list of FastClause protos, which can either
  represent a single negated conjecture (if and_ is specified) or a batch
  of clauses (if and_ is None).

  In the description of the LoomOps below, vocab_id must be VOCAB_ID.

  Args:
    clauses: 1-D `string` tensor of serialized `FastClause` protos.
    embed: LoomOp to embed vocabulary ids: vocab_id -> embedding.
    apply_: LoomOp for curried function application:
        embedding -> embedding -> embedding.
    not_: LoomOp for negation: embedding -> embedding.
    or_: LoomOp for or: embedding -> embedding -> embedding.
    and_: LooOp for and (embedding -> embedding -> embedding) or None if
        clauses is a batch of individual clauses.
    shuffle: Whether to randomly shuffle ands and ors.
    seed: Optional seed for random number generation.

  Returns:
    The final embeddings of each clause, or of the whole conjunction
    if and_ is given.
  """

  def weaver_op(**kwds):
    seed1, seed2 = tf.get_seed(seed)
    return gen_clause_ops.fast_clause_weaver(
        clauses=clauses,
        shuffle=shuffle,
        seed=seed1,
        seed2=seed2,
        conjunction=and_ is not None,
        **kwds)

  ops = {'embed': embed, 'apply': apply_, 'not': not_, 'or': or_}
  if and_ is not None:
    ops['and'] = and_
  clause_loom = loom.Loom(named_ops=ops, weaver_op=weaver_op)
  return tuple(clause_loom.output_tensor(ts) for ts in or_.output_type_shapes)
