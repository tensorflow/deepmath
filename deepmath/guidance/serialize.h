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

#ifndef RESEARCH_MATH_CLAUSE_SEARCH_SERIALIZE_H_
#define RESEARCH_MATH_CLAUSE_SEARCH_SERIALIZE_H_

#include "deepmath/guidance/vocabulary.h"
#include "deepmath/eprover/clause.pb.h"
#include "deepmath/eprover/fol_formula.pb.h"
#include "deepmath/eprover/prover_clause_examples.pb.h"
#include "deepmath/eprover/fast_clause.pb.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"

namespace deepmath {

// There are two different formats for clauses: the ProverClause protos inside
// ProverClauseExamples, and the FastClause protos used to communicate between
// sandbox and sandboxee.  In order to reduce the chance of incompatible
// serialization, we reuse as much code as possible between the two formats.
//
// Specifically, several functions below are templatized over a Context class
// which stores state and parameters for serializing one of the proto types.
// FastSerializeContext is for serializing FastClause(s), and
// SlowSerializeContext is for serializing ProverClause(s).

// Context for serializing FastClause(s)
class FastSerializeContext {
 public:
  FastSerializeContext() = default;

  const std::vector<int>& output() const { return output_; }

  void Add(int id) { output_.push_back(id); }

 private:
  std::vector<int> output_;

  TF_DISALLOW_COPY_AND_ASSIGN(FastSerializeContext);
};

// Context for serializing ProverClause(s)
class SlowSerializeContext {
 public:
  explicit SlowSerializeContext(const Vocabulary& vocab) : vocab_(vocab) {}

  const std::vector<int>& output() const { return output_; }

  void Add(int id) { output_.push_back(id); }
  void Add(const string& word) { output_.push_back(vocab_.Word(word)); }

 private:
  const Vocabulary& vocab_;
  std::vector<int> output_;

  TF_DISALLOW_COPY_AND_ASSIGN(SlowSerializeContext);
};

// Context for serializing FastClause to Vowpal Wabbit format
class VowpalWabbitSerializeContext {
 public:
  explicit VowpalWabbitSerializeContext(const Vocabulary& vocab)
      : vocab_(vocab) {}

  const string& output() const { return output_; }

  void Add(int id) {
    // We substitute 'or' for '|' since '|' has special vowpal wabbit meaning.
    // model.  It'd be nice to eliminate the following hack.
    Add(id == Vocabulary::kOr ? "or" : vocab_.IdToWord(id));
  }

  void Add(const string& word) {
    output_ += word;
    output_.push_back(' ');
  }

 private:
  const Vocabulary& vocab_;
  string output_;

  TF_DISALLOW_COPY_AND_ASSIGN(VowpalWabbitSerializeContext);
};

template <class Context, class Term>
void SerializeArgs(Context* context,
                   const tensorflow::protobuf::RepeatedPtrField<Term>& args) {
  if (!args.empty()) {
    context->Add(Vocabulary::kLeft);
    bool first = true;
    for (const auto& arg : args) {
      if (!first) context->Add(Vocabulary::kComma);
      SerializeTerm(context, arg);
      first = false;
    }
    context->Add(Vocabulary::kRight);
  }
}

template <class Context>
void SerializeTerm(Context* context, const deepmath::FastClause::Term& term) {
  context->Add(term.id());
  SerializeArgs(context, term.args());
}

void SerializeTerm(SlowSerializeContext* context,
                   const deepmath::FirstOrderLogicTerm& term);

template <class Context, class Clause>
void SerializeClause(Context* context, const Clause& clause) {
  if (clause.equations_size() == 0) {
    context->Add(Vocabulary::kFalse);
    return;
  }
  bool first = true;
  for (const auto& equation : clause.equations()) {
    if (!first) context->Add(Vocabulary::kOr);
    if (equation.negated()) context->Add(Vocabulary::kNot);
    SerializeTerm(context, equation.left());
    if (equation.has_right()) {
      context->Add(Vocabulary::kEqual);
      SerializeTerm(context, equation.right());
    }
    first = false;
  }
}

inline void SerializeClause(SlowSerializeContext* context,
                     const ProverClause& clause) {
  SerializeClause(context, clause.clause());
}

// Serialize a conjunction of clauses
template <class Context, class Clauses>
void SerializeClauses(Context* context, const Clauses& clauses) {
  if (clauses.size() == 0) {
    context->Add(Vocabulary::kTrue);
    return;
  }
  bool first = true;
  for (const auto& clause : clauses) {
    if (!first) context->Add(Vocabulary::kAnd);
    context->Add(Vocabulary::kLeft);
    SerializeClause(context, clause);
    context->Add(Vocabulary::kRight);
    first = false;
  }
}

}  // namespace deepmath

#endif  // RESEARCH_MATH_CLAUSE_SEARCH_SERIALIZE_H_
