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

#ifndef RESEARCH_MATH_CLAUSE_SEARCH_VOCABULARY_H_
#define RESEARCH_MATH_CLAUSE_SEARCH_VOCABULARY_H_

#include <unordered_map>
#include <vector>

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace deepmath {

using tensorflow::int32;
using tensorflow::uint32;
using tensorflow::string;

// Manage vocabulary for a clause search model.
class Vocabulary {
 public:
  Vocabulary() = default;

  // Add variables, functions, and numbers from a vocab file.
  //
  // The path can end with :one_variable to use a single vocabulary word
  // for all variables.  Vocabulary files are text files with one vocabulary
  // word per line.
  void Initialize(const tensorflow::StringPiece path);

  // Lookup the id of a vocabulary word
  int32 Word(const string& word) const;

  // The inverse of Word
  const string& IdToWord(int32 id) const {
    CHECK(static_cast<uint32>(id) < static_cast<uint32>(id_to_vocab_.size()));
    return id_to_vocab_[id];
  }

  int32 size() const { return id_to_vocab_.size(); }

  // Constant ids for symbols
  enum {
    kSpace = 0,  //
    kStar = 1,   // *
    kNot = 2,    // ~
    kOr = 3,     // |
    kAnd = 4,    // &
    kLeft = 5,   // (
    kRight = 6,  // )
    kComma = 7,  // ,
    kEqual = 8,  // =
    kFalse = 9,  // $false
    kTrue = 10,  // $true
  };

 private:
  static const int32 kStart = 32;
  std::unordered_map<string, int32> vocab_to_id_;
  std::vector<string> id_to_vocab_;
  int32 variable_id_ = -1;
  bool map_unknown_ = false;

  TF_DISALLOW_COPY_AND_ASSIGN(Vocabulary);
};

}  // namespace deepmath

#endif  // RESEARCH_MATH_CLAUSE_SEARCH_VOCABULARY_H_
