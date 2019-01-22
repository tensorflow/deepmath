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

// Jagged array ops.

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace errors = tensorflow::errors;
namespace internal = tensorflow::internal;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::TTypes;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::int32;

namespace deepmath {
namespace {

REGISTER_OP("Repeats")
    .Input("values: T")
    .Input("times: int32")
    .Output("repeated: T")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle values, times, merged;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &values));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &times));
      TF_RETURN_IF_ERROR(c->Merge(values, times, &merged));
      c->set_output(0, c->UnknownShapeOfRank(1));
      return Status::OK();
    })
    .Doc(R"doc(
Repeat each entry in a vector some number of times.

values: 1-D values to repeat.
times: 1-D numbers of times to repeat each entry.
repeated: 1-D results, with values[i] repeated times[i] times.
)doc");

REGISTER_OP("JaggedMax")
    .Input("sizes: int32")
    .Input("flat: T")
    .Output("max: T")
    .Output("argmax: int32")
    .Attr("T: {float, double}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle sizes, flat, flat_rest, result;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &sizes));
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &flat));
      TF_RETURN_IF_ERROR(c->Subshape(flat, 1, &flat_rest));
      TF_RETURN_IF_ERROR(c->Concatenate(sizes, flat_rest, &result));
      c->set_output(0, result);
      c->set_output(1, result);
      return Status::OK();
    })
    .Doc(R"doc(
Computes componentwise maximum of each sequence in a jagged tensor.

sizes: Jagged.sizes.
flat: Jagged.flat.
max: Componentwise maximum of each jagged[i] sequence.
argmax: Where each maximum is from as an index into the first dimension of flat.
)doc");

template <class T>
class Repeats : public OpKernel {
 public:
  explicit Repeats(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Check input
    const Tensor& values_t = context->input(0);
    const Tensor& times_t = context->input(1);
    OP_REQUIRES(
        context, values_t.dims() == 1 && values_t.shape() == times_t.shape(),
        errors::InvalidArgument(
            "Expected matching 1-D values and times, got shapes ",
            values_t.shape().DebugString(), ", ",
            times_t.shape().DebugString()));
    const auto values = values_t.vec<T>();
    const auto times = times_t.vec<int>();

    // Allocate output
    Eigen::Tensor<int, 0, Eigen::RowMajor> times_sum = times.sum();
    const int total = internal::SubtleMustCopy(times_sum());
    Tensor* output_t = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({total}), &output_t));
    auto output = output_t->vec<T>();

    // Fill output
    for (int i = 0, j = 0; i < values.dimension(0); i++) {
      const int n = internal::SubtleMustCopy(times(i));
      OP_REQUIRES(context, 0 <= n && n <= total - j,
                  errors::InvalidArgument("times[", i, "] = ", n,
                                          " overflows remaining range ", total,
                                          " - ", j, " = ", total - j));
      const auto& value = values(i);
      for (int a = 0; a < n; a++, j++) output(j) = value;
    }
  }
};

template <class T>
class JaggedMax : public OpKernel {
 public:
  explicit JaggedMax(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Check input
    const Tensor& sizes_t = context->input(0);
    const Tensor& flat_t = context->input(1);
    OP_REQUIRES(context, sizes_t.dims() == 1,
                errors::InvalidArgument("Expected 1-D sizes, got shape ",
                                        sizes_t.shape().DebugString()));
    OP_REQUIRES(context, flat_t.dims() >= 1,
                errors::InvalidArgument("flat can't be scalar"));
    const int n = sizes_t.dim_size(0);

    // Allocate output
    TensorShape output_shape = flat_t.shape();
    output_shape.set_dim(0, n);
    Tensor* max_t = nullptr;
    Tensor* argmax_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({output_shape}), &max_t));
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, TensorShape({output_shape}), &argmax_t));

    // Compute maximums
    if (output_shape.num_elements() > 0) {
      const auto sizes = sizes_t.vec<int>();
      const auto flat = flat_t.flat_outer_dims<T>();
      const int limit = flat.dimension(0);
      auto max = max_t->flat_outer_dims<T>();
      auto argmax = argmax_t->flat_outer_dims<int>();
      int prev = 0;
      for (int i = 0; i < n; i++) {
        const int size = internal::SubtleMustCopy(sizes(i));
        const int next = prev + size;
        OP_REQUIRES(
            context, prev < next && next <= limit,
            errors::InvalidArgument(i, "th range [", prev, ",", next,
                                    ") empty or not in [", 0, ",", limit, ")"));
        typedef TTypes<float>::Tensor::Index Index;
        const Eigen::array<Index, 1> first({0});
        const Eigen::DSizes<Index, 2> starts(prev, 0);
        const Eigen::DSizes<Index, 2> slice_sizes(size, flat.dimension(1));
        const auto slice = flat.slice(starts, slice_sizes);
        max.template chip<0>(i) = slice.maximum(first);
        // TODO(geoffreyi): Compute argmax and max together
        const auto relative = slice.argmax(0).template cast<int>();
        argmax.chip<0>(i) = relative.constant(prev) + relative;
        prev = next;
      }
    }
  }
};

#define REGISTER(T)                                                          \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("Repeats").Device(tensorflow::DEVICE_CPU).TypeConstraint<T>("T"), \
      Repeats<T>);
REGISTER(int32)
REGISTER(float)
REGISTER(double)
#undef REGISTER

#define REGISTER(T)                                                            \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("JaggedMax").Device(tensorflow::DEVICE_CPU).TypeConstraint<T>("T"), \
      JaggedMax<T>);
REGISTER(float)
REGISTER(double)

}  // namespace
}  // namespace deepmath
