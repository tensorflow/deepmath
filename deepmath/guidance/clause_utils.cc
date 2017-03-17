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

#include "deepmath/guidance/clause_utils.h"

#include "tensorflow/core/lib/random/simple_philox.h"

namespace errors = tensorflow::errors;
namespace protobuf = tensorflow::protobuf;
using tensorflow::OpKernelConstruction;
using tensorflow::Status;
using tensorflow::uint8;

namespace deepmath {

static const int kRecursionLimit = 10000;

RandomClausesHelper::RandomClausesHelper(OpKernelConstruction* context) {
  OP_REQUIRES_OK(context, generator_.Init(context));
  string vocab;
  OP_REQUIRES_OK(context, context->GetAttr("vocab", &vocab));
  vocab_.Initialize(vocab);
}

bool RandomClausesHelper::Choose(const ProverClauseExamples& examples,
                                 const ProverClause** positive,
                                 const ProverClause** negative) {
  if (!examples.positives_size() || !examples.negatives_size()) return false;
  auto gen = generator_.ReserveSamples32(2 * 256);
  tensorflow::random::SimplePhilox random(&gen);
  *positive = &examples.positives(random.Uniform(examples.positives_size()));
  *negative = &examples.negatives(random.Uniform(examples.negatives_size()));
  return true;
}

Status ParseExamples(const tensorflow::StringPiece proto,
                     ProverClauseExamples* examples) {
  protobuf::io::CodedInputStream stream(
      reinterpret_cast<const uint8*>(proto.data()), proto.size());
  stream.SetRecursionLimit(kRecursionLimit);
  if (!examples->ParseFromCodedStream(&stream)) {
    return errors::InvalidArgument(
        "Can't parse examples proto as ProverClauseExamples");
  }
  return Status::OK();
}

Status ParseFastClause(const tensorflow::StringPiece proto,
                       FastClause* clause) {
  protobuf::io::CodedInputStream stream(
      reinterpret_cast<const uint8*>(proto.data()), proto.size());
  stream.SetRecursionLimit(kRecursionLimit);
  if (!clause->ParseFromCodedStream(&stream)) {
    return errors::InvalidArgument("Failed to parse FastClause");
  }
  return Status::OK();
}

}  // namespace deepmath
