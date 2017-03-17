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

#ifndef RESEARCH_MATH_CLAUSE_SEARCH_CLAUSE_UTILS_H_
#define RESEARCH_MATH_CLAUSE_SEARCH_CLAUSE_UTILS_H_

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/guarded_philox_random.h"

#include "deepmath/guidance/vocabulary.h"
#include "deepmath/eprover/prover_clause_examples.pb.h"
#include "deepmath/eprover/fast_clause.pb.h"

namespace deepmath {

class RandomClausesHelper {
 public:
  explicit RandomClausesHelper(tensorflow::OpKernelConstruction* context);

  // Pick a random positive and negative, or return false if either is
  // unavailable.
  bool Choose(const ProverClauseExamples& examples,
              const ProverClause** positive, const ProverClause** negative);

  tensorflow::GuardedPhiloxRandom& generator() { return generator_; }
  const Vocabulary& vocab() const { return vocab_; }

 private:
  tensorflow::GuardedPhiloxRandom generator_;
  Vocabulary vocab_;
};

tensorflow::Status ParseExamples(const tensorflow::StringPiece proto,
                                 ProverClauseExamples* examples);

tensorflow::Status ParseFastClause(const tensorflow::StringPiece proto,
                                   deepmath::FastClause* clause);

}  // namespace deepmath

#endif  // RESEARCH_MATH_CLAUSE_SEARCH_CLAUSE_UTILS_H_
