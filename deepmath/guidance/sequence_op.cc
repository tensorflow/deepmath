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

// Serialize clause examples as sequences

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "deepmath/guidance/clause_utils.h"
#include "deepmath/guidance/make_fast_clause.h"
#include "deepmath/guidance/serialize.h"
#include "deepmath/guidance/vocabulary.h"
#include "deepmath/eprover/clause.pb.h"
#include "deepmath/eprover/fol_formula.pb.h"
#include "deepmath/eprover/prover_clause_examples.pb.h"

using deepmath::FastClause;
using deepmath::FirstOrderLogicTerm;
using deepmath::FirstOrderLogicClause;
namespace errors = tensorflow::errors;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::TensorShapeUtils;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::int64;

namespace deepmath {
namespace {

REGISTER_OP("RandomClausesAsSequence")
    .Input("examples: string")
    .Output("key: string")
    .Output("negated_conjecture: string")
    .Output("clauses: string")
    .Output("labels: bool")
    .Attr("vocab: string")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Doc(R"doc(
Select one positive and one negative clause from a ProverClauseExamples.

If the ProverClauseExamples doesn't have both positives and negatives,
clauses and labels will be empty.

examples: Serialized ProverClauseExamples.
key: Key field of ProverClauseExamples.
negated_conjecture: 0-D `int32` negated conjecture encoded as `string`.
clauses: 1-D `int32` clauses encoded as `string`: one positive, one negative.
labels: 1-D labels (true for positive, false for negative).
vocab: Path to vocabulary file.
seed: If either `seed` or `seed2` are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: A second seed to avoid seed collision.
)doc");

REGISTER_OP("FastClausesAsSequence")
    .Input("fast_clauses: string")
    .Output("ids: int32")
    .Attr("conjunction: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle fast_clauses;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &fast_clauses));
      bool conjunction;
      TF_RETURN_IF_ERROR(c->GetAttr("conjunction", &conjunction));
      if (conjunction) {
        c->set_output(0, c->UnknownShapeOfRank(1));
      } else {
        c->set_output(0, c->Matrix(c->Dim(fast_clauses, 0), c->UnknownDim()));
      }
      return Status::OK();
    })
    .Doc(R"doc(
Serialize FastClauses as sequences of ids.

fast_clauses: Serialized FastClause protos.
conjunction: Whether to interpret the last dimension as a conjunction.
ids: Vocab id sequences, padded on the right.
)doc");

REGISTER_OP("FastClausesAsSequenceJagged")
    .Input("fast_clauses: string")
    .Output("sizes: int32")
    .Output("flat: int32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle fast_clauses;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &fast_clauses));
      c->set_output(0, fast_clauses);
      c->set_output(1, c->UnknownShapeOfRank(1));
      return Status::OK();
    })
    .Doc(R"doc(
Serialize FastClauses as sequences of ids.

fast_clauses: Serialized FastClause protos.
sizes: Length of each clause.
flat: Concatenated tokens ids from each clause.
)doc");

class RandomClausesAsSequence : public OpKernel {
 public:
  explicit RandomClausesAsSequence(OpKernelConstruction* context)
      : OpKernel(context), helper_(context) {}

  void Compute(OpKernelContext* context) override {
    // Parse and choose positive and negative
    const Tensor& serialized = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(serialized.shape()),
                errors::InvalidArgument("Need scalar examples, got shape ",
                                        serialized.shape().DebugString()));
    ProverClauseExamples examples;
    OP_REQUIRES_OK(context,
                   ParseExamples(serialized.scalar<string>()(), &examples));
    const ProverClause *positive, *negative;
    const bool valid = helper_.Choose(examples, &positive, &negative);

    // Allocate outputs
    Tensor* key = nullptr;
    Tensor* negated_conjecture = nullptr;
    Tensor* clauses = nullptr;
    Tensor* labels = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &key));
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}),
                                                     &negated_conjecture));
    OP_REQUIRES_OK(context, context->allocate_output(
                                2, TensorShape({2 * valid}), &clauses));
    OP_REQUIRES_OK(context, context->allocate_output(
                                3, TensorShape({2 * valid}), &labels));

    // Grab key
    key->scalar<string>()() = examples.key();

    // Serialize negative conjecture
    SerializeToString(&SerializeClauses,
                      examples.cnf().negated_conjecture(),
                      &negated_conjecture->scalar<string>()());

    // If possible, serialize positive and negative examples
    if (valid) {
      SerializeToString(&SerializeClause, positive->clause(),
                        &clauses->vec<string>()(0));
      SerializeToString(&SerializeClause, negative->clause(),
                        &clauses->vec<string>()(1));
      labels->vec<bool>()(0) = true;
      labels->vec<bool>()(1) = false;
    }
  }

 private:
  RandomClausesHelper helper_;

  template <class T>
  void SerializeToString(void (*serialize)(SlowSerializeContext*, const T&),
                         const T& data, string* output) {
    SlowSerializeContext context(helper_.vocab());
    serialize(&context, data);
    output->insert(0, reinterpret_cast<const char*>(context.output().data()),
                   context.output().size() * sizeof(int));
  }
};

class FastClausesAsSequence : public OpKernel {
 public:
  explicit FastClausesAsSequence(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("conjunction", &conjunction_));
  }

  void Compute(OpKernelContext* context) override {
    // Check input
    const Tensor& protos = context->input(0);
    OP_REQUIRES(context, protos.dims() == 1,
                errors::InvalidArgument("Expected 1-D input, got ",
                                        protos.shape().DebugString()));
    const auto protos_vec = protos.vec<string>();
    const int64 n = protos_vec.dimension(0);

    if (conjunction_) {
      // Parse all clauses
      std::vector<FastClause> clauses(n);
      for (int64 i = 0; i < n; i++) {
        OP_REQUIRES_OK(context, ParseFastClause(protos_vec(i), &clauses[i]));
      }
      // Convert to id sequence
      FastSerializeContext fast;
      SerializeClauses(&fast, clauses);
      const std::vector<int>& sequence = fast.output();
      const int64 length = sequence.size();
      // Copy to output
      Tensor* output = nullptr;
      OP_REQUIRES_OK(
          context, context->allocate_output(0, TensorShape({length}), &output));
      memcpy(output->vec<int>().data(), sequence.data(), length * sizeof(int));
    } else {
      // Parse all clauses and convert to sequences
      std::vector<FastSerializeContext> fasts(n);
      int64 max_size = 0;
      {
        FastClause clause;
        for (int64 i = 0; i < n; i++) {
          OP_REQUIRES_OK(context, ParseFastClause(protos_vec(i), &clause));
          SerializeClause(&fasts[i], clause);
          max_size =
              std::max(max_size, static_cast<int64>(fasts[i].output().size()));
        }
      }
      // Pad and copy to output
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(
                                  0, TensorShape({n, max_size}), &output));
      auto output_flat = output->flat<int>();
      output_flat.setZero();
      for (int64 i = 0; i < n; i++) {
        const std::vector<int>& seq = fasts[i].output();
        memcpy(output_flat.data() + i * max_size, seq.data(),
               seq.size() * sizeof(int));
      }
    }
  }

 private:
  bool conjunction_;
};

class FastClausesAsSequenceJagged : public OpKernel {
 public:
  explicit FastClausesAsSequenceJagged(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Check input
    const Tensor& protos_t = context->input(0);
    OP_REQUIRES(context, protos_t.dims() == 1,
                errors::InvalidArgument("Expected 1-D input, got ",
                                        protos_t.shape().DebugString()));
    const auto protos = protos_t.vec<string>();

    // Allocate sizes
    Tensor* sizes_t = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({protos.dimension(0)}),
                                          &sizes_t));
    auto sizes = sizes_t->vec<int>();

    // Parse all clauses and convert to sequences
    FastSerializeContext fast;
    {
      FastClause clause;
      for (int64 i = 0; i < protos.dimension(0); i++) {
        OP_REQUIRES_OK(context, ParseFastClause(protos(i), &clause));
        const int before = fast.output().size();
        SerializeClause(&fast, clause);
        sizes(i) = fast.output().size() - before;
      }
    }

    // Copy flat to output
    Tensor* flat_t = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(
                     1, TensorShape({static_cast<int64>(fast.output().size())}),
                     &flat_t));
    auto flat = flat_t->vec<int>();
    memcpy(flat.data(), fast.output().data(),
           fast.output().size() * sizeof(int));
  }
};

REGISTER_KERNEL_BUILDER(
    Name("RandomClausesAsSequence").Device(tensorflow::DEVICE_CPU),
    RandomClausesAsSequence);

REGISTER_KERNEL_BUILDER(
    Name("FastClausesAsSequence").Device(tensorflow::DEVICE_CPU),
    FastClausesAsSequence);

REGISTER_KERNEL_BUILDER(
    Name("FastClausesAsSequenceJagged").Device(tensorflow::DEVICE_CPU),
    FastClausesAsSequenceJagged);

}  // namespace
}  // namespace deepmath
