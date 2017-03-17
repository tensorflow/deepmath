/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// TensorLoom weaver for ProverClauseExamples

#include "tensorflow_fold/public/loom.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/logging.h"

#include "deepmath/guidance/clause_utils.h"
#include "deepmath/guidance/vocabulary.h"
#include "deepmath/eprover/clause.pb.h"
#include "deepmath/eprover/fol_formula.pb.h"
#include "deepmath/eprover/prover_clause_examples.pb.h"

using deepmath::FastClause;
using deepmath::FirstOrderLogicTerm;
namespace errors = tensorflow::errors;
using tensorflow::GuardedPhiloxRandom;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::TensorShapeUtils;
using tensorflow::gtl::ArraySlice;
using tensorflow::fold::Weaver;
using tensorflow::fold::WeaverOpBase;
using tensorflow::int64;

namespace deepmath {
namespace {

REGISTER_WEAVER_OP("RandomClausesWeaver")
    .Input("examples: string")
    .Attr("vocab: string")
    .Attr("shuffle: bool")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Doc(R"doc(
TensorLoom weaver for ProverClauseExamples protos.

Computes results for a batch of ProverClauseExamples protos.  For each
ProverClauseExamples with at least one positive and negative, one positive
and one negative are selected and reduced to results according to the following
loom ops:

    # Embedding layer shared between conjecture and clause
    embed : vocab_id -> embedding

    # Tree embedding for x in (conjecture, clause)
    x_apply : function_embedding -> arg_embedding -> embedding
    x_not : embedding -> embedding
    x_or : left_embedding -> right_embedding -> embedding
    x_and : left_embedding -> right_embedding -> embedding  # conjecture only

    # Combine conjectures and clauses
    combine : conjecture_embedding -> clause_embedding -> combination

There are two output type shapes: combination and label.  The type shapes
`vocab_id` and `label` must be scalar `int32` and `bool`, respectively.

examples: 1-D batch of serialized ProverClauseExamples.
vocab: Path to vocabulary file.
shuffle: If true, randomize the ordering of disjunctions and conjuctions before
  reducing with x_or / x_and.
seed: If either `seed` or `seed2` are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: A second seed to avoid seed collision.
)doc");

REGISTER_WEAVER_OP("FastClauseWeaver")
    .Input("clauses: string")
    .Attr("conjunction: bool = false")
    .Attr("shuffle: bool")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Doc(R"doc(
TensorLoom weaver for FastClause protos.

Computes embeddings for a batch of FastClause protos, interpreted either as a
single negated conjecture (conjunction = true) or a batch of individual clauses
(conjunction = false).  The clauses are reduced to embeddings according to the
following loom ops:

    # Embedding layer shared between conjecture and clause
    embed : vocab_id -> embedding

    # Tree embedding for x in (conjecture, clause)
    apply : function_embedding -> arg_embedding -> embedding
    not : embedding -> embedding
    or : left_embedding -> right_embedding -> embedding
    and : left_embedding -> right_embedding -> embedding  # conjunction only

clauses: 1-D batch of serialized FastClause protos.
conjunction: Whether to interpret the batch as a conjunction or as individual
  protos.
shuffle: If true, randomize the ordering of disjunctions and conjuctions before
  reducing with or / and.
seed: If either `seed` or `seed2` are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: A second seed to avoid seed collision.
)doc");

// Wrap the ops used for clause embedding in a struct so that we can have one
// copy for negated conjecture and one for the clause to be evaluated.
struct ClauseOps {
  int32 apply_op, not_op, or_op, and_op;
};

typedef std::vector<int32> Values;

template <class Embed>
struct Context {
  const ClauseOps ops;
  GuardedPhiloxRandom* const generator;
  const Embed embed;
  Weaver* const weaver;
  const bool shuffle;
  Values false_id, true_id, eq_id;
};

Values Flatten(const std::initializer_list<Values>& values) {
  Values flat;
  for (const auto& v : values) {
    flat.insert(flat.end(), v.begin(), v.end());
  }
  return flat;
}

template <class T>
int32 WeaveScalar(int32 type_shape, const T scalar, Weaver* weaver) {
  static_assert(std::is_integral<T>::value, "Expected integral type");
  string bytes(reinterpret_cast<const char*>(&scalar), sizeof(T));
  int32 result = weaver->MakeConstantSerialized(type_shape, bytes);
  CHECK_GE(result, 0) << weaver->error_string();
  return result;
}

// Combine entries using a random complete binary tree.
template <class Context>
Values ShuffleWeaveList(const Context& context, const int32 op,
                        const Values& empty, ArraySlice<Values> entries) {
  if (entries.empty()) return empty;
  std::vector<Values> prev(entries.begin(), entries.end()), next;
  if (context.shuffle && entries.size() > 1) {
    auto gen = context.generator->ReserveSamples32(256 * entries.size());
    tensorflow::random::SimplePhilox random(&gen);
    std::random_shuffle(prev.begin(), prev.end(),
                        [&random](int n) { return random.Uniform(n); });
  }
  while (prev.size() > 1) {
    for (int i = 0; i + 1 < prev.size(); i += 2) {
      next.push_back(
          context.weaver->CallOp(op, Flatten({prev[i], prev[i + 1]})));
    }
    if (prev.size() & 1) next.push_back(prev.back());
    prev.swap(next);
    next.clear();
  }
  return prev[0];
}

template <class Context, class Args>
Values WeaveApply(const Context& context, const Values& f, const Args& args) {
  Values state = f;
  for (const auto& arg : args) {
    state = context.weaver->CallOp(context.ops.apply_op,
                                   Flatten({state, WeaveTerm(context, arg)}));
  }
  return state;
}

template <class Context>
Values WeaveTerm(const Context& context, const FastClause::Term& term) {
  return WeaveApply(context, context.embed(term.id()), term.args());
}

template <class Context>
Values WeaveTerm(const Context& context, const FirstOrderLogicTerm& term) {
  switch (term.term_type_case()) {
    case FirstOrderLogicTerm::kVariable:
      return context.embed(term.variable().name());
    case FirstOrderLogicTerm::kNumber:
      return context.embed(term.number().value());
    case FirstOrderLogicTerm::kFunction: {
      auto& f = term.function();
      return WeaveApply(context, context.embed(f.name()), f.args());
    }
    default:
      LOG(FATAL) << "Invalid term type " << term.term_type_case();
  }
}

template <class Context, class Clause>
Values WeaveClause(const Context& context, const Clause& clause) {
  std::vector<Values> list;
  for (const auto& equation : clause.equations()) {
    typedef
        typename std::remove_reference<decltype(equation.left())>::type Term;
    std::reference_wrapper<Term> args[2] = {equation.left(), equation.right()};
    Values eqn = equation.has_right() ? WeaveApply(context, context.eq_id, args)
                                      : WeaveTerm(context, args[0]);
    if (equation.negated()) {
      eqn = context.weaver->CallOp(context.ops.not_op, eqn);
    }
    list.push_back(eqn);
  }
  return ShuffleWeaveList(context, context.ops.or_op, context.false_id, list);
}

template <class Context>
Values WeaveClause(const Context& context, const ProverClause& clause) {
  return WeaveClause(context, clause.clause());
}

template <class Context, class Clauses>
Values WeaveClauses(const Context& context, const Clauses& clauses) {
  std::vector<Values> list;
  for (const auto& clause : clauses) {
    list.push_back(WeaveClause(context, clause));
  }
  return ShuffleWeaveList(context, context.ops.and_op, context.true_id, list);
}

class RandomClausesWeaver : public WeaverOpBase {
 public:
  explicit RandomClausesWeaver(OpKernelConstruction* context)
      : WeaverOpBase(context), helper_(context) {
    OP_REQUIRES_OK(context, context->GetAttr("shuffle", &shuffle_));
    OP_REQUIRES_OK(context, FindTypeShape("vocab_id", &vocab_id_ts_));
    OP_REQUIRES_OK(context, FindTypeShape("label", &label_ts_));
    OP_REQUIRES_OK(context, FindOp("embed", &embed_op_));
    OP_REQUIRES_OK(context,
                   FindOp("conjecture_apply", &conjecture_ops_.apply_op));
    OP_REQUIRES_OK(context, FindOp("conjecture_not", &conjecture_ops_.not_op));
    OP_REQUIRES_OK(context, FindOp("conjecture_or", &conjecture_ops_.or_op));
    OP_REQUIRES_OK(context, FindOp("conjecture_and", &conjecture_ops_.and_op));
    OP_REQUIRES_OK(context, FindOp("clause_apply", &clause_ops_.apply_op));
    OP_REQUIRES_OK(context, FindOp("clause_not", &clause_ops_.not_op));
    OP_REQUIRES_OK(context, FindOp("clause_or", &clause_ops_.or_op));
    OP_REQUIRES_OK(context, FindOp("combine", &combine_op_));
    clause_ops_.and_op = -1;
  }

  Status Weave(OpKernelContext* context, Weaver* weaver) override {
    // Validate
    const Tensor& serialized = context->input(0);
    if (!TensorShapeUtils::IsVector(serialized.shape())) {
      return errors::InvalidArgument("Expected 1-D examples, got shape ",
                                     serialized.shape().DebugString());
    }

    // Prepare to weave
    std::unordered_map<string, Values> vocab_map;
    const auto serialized_vec = serialized.vec<string>();
    auto embed = [=, &vocab_map](const string& word) {
      // Return quickly if we've already looked up this embedding
      auto it = vocab_map.find(word);
      if (it != vocab_map.end()) return it->second;

      // Lookup the embedding and cache it
      const int64 id = helper_.vocab().Word(word);
      auto result =
          weaver->CallOp(embed_op_, {WeaveScalar(vocab_id_ts_, id, weaver)});
      vocab_map.emplace(word, result);
      return result;
    };
    Context<decltype(embed)> conjecture({conjecture_ops_, &helper_.generator(),
                                         embed, weaver, shuffle_});
    Context<decltype(embed)> clause({clause_ops_, &helper_.generator(), embed,
                                     weaver, shuffle_});
    conjecture.true_id = embed("$true");
    conjecture.false_id = clause.false_id = embed("$false");
    conjecture.eq_id = clause.eq_id = embed("=");

    // Weave each ProverClauseExamples with at least one positive and negative
    ProverClauseExamples examples;
    for (int64 i = 0; i < serialized_vec.dimension(0); i++) {
      TF_RETURN_IF_ERROR(ParseExamples(serialized_vec(i), &examples));
      const ProverClause *positive, *negative;
      if (helper_.Choose(examples, &positive, &negative)) {
        const auto conjecture_embedding =
            WeaveClauses(conjecture, examples.cnf().negated_conjecture());
        const auto positive_embedding = WeaveClause(clause, *positive);
        const auto negative_embedding = WeaveClause(clause, *negative);
        for (const bool label : {true, false}) {
          auto clause_emb = label ? positive_embedding : negative_embedding;
          for (const int32 output : weaver->CallOp(
                   combine_op_, Flatten({conjecture_embedding, clause_emb}))) {
            weaver->AddOutput(output);
          }
          weaver->AddOutput(WeaveScalar(label_ts_, label, weaver));
        }
      }
    }
    return Status::OK();
  }

 private:
  RandomClausesHelper helper_;
  bool shuffle_;
  int32 vocab_id_ts_, label_ts_;  // TypeShapes for vocab ids and labels

  // Loom op ids.  For details, see the op doc string.
  int32 embed_op_, combine_op_;
  ClauseOps conjecture_ops_, clause_ops_;
};

class FastClauseWeaver : public WeaverOpBase {
 public:
  explicit FastClauseWeaver(OpKernelConstruction* context)
      : WeaverOpBase(context) {
    OP_REQUIRES_OK(context, generator_.Init(context));
    OP_REQUIRES_OK(context, context->GetAttr("conjunction", &conjunction_));
    OP_REQUIRES_OK(context, context->GetAttr("shuffle", &shuffle_));
    OP_REQUIRES_OK(context, FindTypeShape("vocab_id", &vocab_id_ts_));
    OP_REQUIRES_OK(context, FindOp("embed", &embed_op_));
    OP_REQUIRES_OK(context, FindOp("apply", &ops_.apply_op));
    OP_REQUIRES_OK(context, FindOp("not", &ops_.not_op));
    OP_REQUIRES_OK(context, FindOp("or", &ops_.or_op));
    if (conjunction_) {
      OP_REQUIRES_OK(context, FindOp("and", &ops_.and_op));
    } else {
      ops_.and_op = -1;
    }
  }

  Status Weave(OpKernelContext* context, Weaver* weaver) override {
    // Validate
    const Tensor& protos = context->input(0);
    if (!TensorShapeUtils::IsVector(protos.shape())) {
      return errors::InvalidArgument("Expected 1-D examples, got shape ",
                                     protos.shape().DebugString());
    }
    const auto protos_vec = protos.vec<string>();

    // Prepare to weave
    std::unordered_map<int, Values> vocab_map;
    auto embed = [=, &vocab_map](const int id) {
      // Return quickly if we've already looked up this embedding
      auto it = vocab_map.find(id);
      if (it != vocab_map.end()) return it->second;

      // Lookup the embedding and cache it
      const int64 big = id;
      auto result =
          weaver->CallOp(embed_op_, {WeaveScalar(vocab_id_ts_, big, weaver)});
      vocab_map.emplace(id, result);
      return result;
    };
    Context<decltype(embed)> helper({ops_, &generator_, embed, weaver,
                                     shuffle_});
    helper.eq_id = embed(Vocabulary::kEqual);
    helper.false_id = embed(Vocabulary::kFalse);

    if (conjunction_) {
      // Embed all the clauses as one negated conjecture
      helper.true_id = embed(Vocabulary::kTrue);
      std::vector<FastClause> clauses(protos_vec.dimension(0));
      for (int i = 0; i < clauses.size(); i++) {
        TF_RETURN_IF_ERROR(ParseFastClause(protos_vec(i), &clauses[i]));
      }
      for (auto output : WeaveClauses(helper, clauses)) {
        weaver->AddOutput(output);
      }
    } else {
      // Embed each clause separately
      FastClause clause;
      for (int i = 0; i < protos_vec.dimension(0); i++) {
        TF_RETURN_IF_ERROR(ParseFastClause(protos_vec(i), &clause));
        for (auto output : WeaveClause(helper, clause)) {
          weaver->AddOutput(output);
        }
      }
    }
    return Status::OK();
  }

 private:
  GuardedPhiloxRandom generator_;
  bool conjunction_;
  bool shuffle_;
  int32 vocab_id_ts_;
  int32 embed_op_;
  ClauseOps ops_;
};

REGISTER_KERNEL_BUILDER(
    Name("RandomClausesWeaver").Device(tensorflow::DEVICE_CPU),
    RandomClausesWeaver);

REGISTER_KERNEL_BUILDER(Name("FastClauseWeaver").Device(tensorflow::DEVICE_CPU),
                        FastClauseWeaver);

}  // namespace
}  // namespace deepmath
