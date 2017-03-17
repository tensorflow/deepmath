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

// Convert between different clause protos

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "deepmath/guidance/clause_utils.h"
#include "deepmath/guidance/make_fast_clause.h"
#include "deepmath/guidance/vocabulary.h"
#include "deepmath/eprover/prover_clause_examples.pb.h"

namespace errors = tensorflow::errors;
namespace protobuf = tensorflow::protobuf;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

namespace deepmath {
namespace {

REGISTER_OP("ClausesAsFastClause")
    .Input("examples: string")
    .Output("negated_conjecture: string")
    .Output("positives: string")
    .Output("negatives: string")
    .Attr("vocab: string")
    .Doc(R"doc(
Convert all clauses in a ProverClauseExamples to FastClause protos.

examples: Serialized ProverClauseExamples.
negated_conjecture: 1-D negated conjecture encoded as `FastClause`.
positives: 1-D positive clauses as `FastClause`.
negatives: 1-D positive clauses as `FastClause`.
vocab: Path to vocabulary file.
)doc");

REGISTER_OP("RandomClausesAsFastClause")
    .Input("examples: string")
    .Output("conjecture_sizes: int32")
    .Output("conjecture_flat: string")
    .Output("clauses: string")
    .Output("labels: bool")
    .Attr("vocab: string")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle examples;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &examples));
      const auto batch = c->Dim(examples, 0);
      DimensionHandle two_batch;
      TF_RETURN_IF_ERROR(c->Multiply(batch, 2, &two_batch));
      for (int i = 0; i < 4; i++) {
        c->set_output(i, c->UnknownShapeOfRank(1));
      }
      return Status::OK();
    })
    .Doc(R"doc(
Convert one positive and negative from each ProverClauseExamples to FastClause.

If a ProverClauseExamples doesn't have both positives and negatives, it will
be skipped.

The number of clauses in each conjecture is recorded in conjecture_sizes, and
all clauses from all conjectures are concatenated into conjecture_flat.

examples: 1-D serialized ProverClauseExamples protos.
conjecture_sizes: Number of clauses in each conjecture.
conjecture_flat: Conjecture clauses concatenated together, as FastClause protos.
clauses: Positive and negative clauses corresponding to conjectures, interleaved
  as FastClause protos.
seed: If either `seed` or `seed2` are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: A second seed to avoid seed collision.
)doc");

class ClausesAsFastClause : public OpKernel {
 public:
  explicit ClausesAsFastClause(OpKernelConstruction* context)
      : OpKernel(context) {
    string vocab;
    OP_REQUIRES_OK(context, context->GetAttr("vocab", &vocab));
    vocab_.Initialize(vocab);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& proto = context->input(0);
    OP_REQUIRES(context, proto.dims() == 0,
                errors::InvalidArgument("Need scalar examples, got shape ",
                                        proto.shape().DebugString()));
    ProverClauseExamples examples;
    OP_REQUIRES_OK(context, ParseExamples(proto.scalar<string>()(), &examples));
    OP_REQUIRES_OK(context,
                   Output(context, 0, examples.cnf().negated_conjecture()));
    OP_REQUIRES_OK(context, Output(context, 1, examples.positives()));
    OP_REQUIRES_OK(context, Output(context, 2, examples.negatives()));
  }

 private:
  Status Output(OpKernelContext* context, const int which,
                const protobuf::RepeatedPtrField<ProverClause>& clauses) {
    // Allocate output
    Tensor* output = nullptr;
    TF_RETURN_IF_ERROR(context->allocate_output(
        which, TensorShape({clauses.size()}), &output));

    // Vocabulary lookup
    const auto lookup = [this](const string& word) {
      return vocab_.Word(word);
    };

    // Convert negative conjecture
    deepmath::FastClause clause;
    auto output_vec = output->vec<string>();
    for (int i = 0; i < clauses.size(); i++) {
      MakeFastClause(clauses.Get(i), lookup, &clause);
      CHECK(clause.SerializeToString(&output_vec(i)));
    }
    return Status::OK();
  }

  Vocabulary vocab_;
};

class RandomClausesAsFastClause : public OpKernel {
 public:
  explicit RandomClausesAsFastClause(OpKernelConstruction* context)
      : OpKernel(context), helper_(context) {}

  void Compute(OpKernelContext* context) override {
    // Check input
    const Tensor& protos_t = context->input(0);
    OP_REQUIRES(context, protos_t.dims() == 1,
                errors::InvalidArgument("Need 1-D examples, got shape ",
                                        protos_t.shape().DebugString()));
    const auto protos = protos_t.vec<string>();

    // Vocabulary lookup
    const auto lookup = [this](const string& word) {
      return helper_.vocab().Word(word);
    };

    // Convert to FastClause
    struct Entry {
      std::vector<string> conjecture;
      string positive, negative;
    };
    std::vector<Entry> entries;
    int conjecture_total = 0;
    {
      ProverClauseExamples examples;
      deepmath::FastClause fast;
      for (int i = 0; i < protos.dimension(0); i++) {
        OP_REQUIRES_OK(context, ParseExamples(protos(i), &examples));
        const ProverClause *positive, *negative;
        if (helper_.Choose(examples, &positive, &negative)) {
          Entry entry;
          for (const auto& clause : examples.cnf().negated_conjecture()) {
            string s;
            MakeFastClause(clause, lookup, &fast);
            CHECK(fast.SerializeToString(&s));
            entry.conjecture.push_back(s);
          }
          conjecture_total += entry.conjecture.size();
          MakeFastClause(*positive, lookup, &fast);
          CHECK(fast.SerializeToString(&entry.positive));
          MakeFastClause(*negative, lookup, &fast);
          CHECK(fast.SerializeToString(&entry.negative));
          entries.emplace_back(std::move(entry));
        }
      }
    }

    // Allocate outputs
    Tensor* conjecture_sizes_t = nullptr;
    Tensor* conjecture_flat_t = nullptr;
    Tensor* clauses_t = nullptr;
    Tensor* labels_t = nullptr;
    const int32 num_entries = entries.size();
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({num_entries}),
                                            &conjecture_sizes_t));
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape({conjecture_total}),
                                            &conjecture_flat_t));
    OP_REQUIRES_OK(context, context->allocate_output(
                                2, TensorShape({2 * num_entries}), &clauses_t));
    OP_REQUIRES_OK(context, context->allocate_output(
                                3, TensorShape({2 * num_entries}), &labels_t));
    if (entries.empty()) return;

    // Fill outputs
    auto conjecture_sizes = conjecture_sizes_t->vec<int>();
    auto conjecture_flat = conjecture_flat_t->vec<string>();
    auto clauses = clauses_t->vec<string>();
    auto labels = labels_t->vec<bool>();
    for (int i = 0, j = 0; i < num_entries; i++) {
      const auto& entry = entries[i];
      conjecture_sizes(i) = entry.conjecture.size();
      for (const auto& s : entry.conjecture) conjecture_flat(j++) = s;
      clauses(2 * i + 0) = entry.positive;
      clauses(2 * i + 1) = entry.negative;
      labels(2 * i + 0) = true;
      labels(2 * i + 1) = false;
    }
  }

 private:
  RandomClausesHelper helper_;
};

REGISTER_KERNEL_BUILDER(
    Name("ClausesAsFastClause").Device(tensorflow::DEVICE_CPU),
    ClausesAsFastClause);

REGISTER_KERNEL_BUILDER(
    Name("RandomClausesAsFastClause").Device(tensorflow::DEVICE_CPU),
    RandomClausesAsFastClause);

}  // namespace
}  // namespace deepmath
