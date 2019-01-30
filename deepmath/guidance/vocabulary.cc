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

#include "deepmath/guidance/vocabulary.h"

#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"

namespace deepmath {

using tensorflow::StringPiece;

static bool is_variable(const StringPiece word) {
  return !word.empty() && isupper(word[0]);
}

// Add variables, functions, and numbers from a file
void Vocabulary::Initialize(tensorflow::StringPiece path) {
  QCHECK(id_to_vocab_.empty()) << "Initialize called twice";

  // Remove flags from path
  bool one_variable = false;
  const auto slash = path.rfind('/');
  const auto colon = path.rfind(':');
  if (colon != string::npos && (slash == string::npos || slash < colon)) {
    for (const auto& flag :
         tensorflow::str_util::Split(path.substr(colon + 1), ',')) {
      if (flag == "one_variable") {
        one_variable = true;
      } else if (flag == "map_unknown") {
        map_unknown_ = true;
      } else {
        LOG(FATAL) << "Invalid vocabulary flag '" << flag << "'";
      }
    }
    path = path.substr(0, colon);
  }

  // A few symbols are used in string form by the weaver
  id_to_vocab_.resize(kStart);
  id_to_vocab_[kEqual] = "=";
  id_to_vocab_[kFalse] = "$false";
  id_to_vocab_[kTrue] = "$true";

  // Load vocabulary
  {
    string all_vocab;
    TF_CHECK_OK(tensorflow::ReadFileToString(tensorflow::Env::Default(),
                                             string(path), &all_vocab));
    for (const auto& raw_word : tensorflow::str_util::Split(all_vocab, '\n')) {
      tensorflow::StringPiece word = raw_word;
      tensorflow::str_util::RemoveWhitespaceContext(&word);
      if (word.empty()) continue;
      for (const auto c : word) {
        QCHECK(isalnum(c) || c == '_') << "Nonalphanumeric vocab word '"
            << tensorflow::str_util::CEscape(word) << " in " << path;
      }
      if (one_variable && is_variable(word)) {
        if (variable_id_ < 0) {
          variable_id_ = id_to_vocab_.size();
          id_to_vocab_.push_back("X");
        }
      } else {
        id_to_vocab_.emplace_back(word);
      }
    }
  }

  // Invert id_to_vocab_
  for (int i = 0; i < id_to_vocab_.size(); i++) {
    if (!id_to_vocab_[i].empty()) {
      vocab_to_id_.emplace(id_to_vocab_[i], i);
    }
  }
}

// Lookup the id of a vocabulary word
int32 Vocabulary::Word(const string& word) const {
  if (variable_id_ >= 0 && is_variable(word)) return variable_id_;
  auto it = vocab_to_id_.find(word);
  if (it == vocab_to_id_.end() && map_unknown_) {
    if (variable_id_ >= 0)
      return variable_id_;
    else
      it = vocab_to_id_.find("X49");
  }
  QCHECK(it != vocab_to_id_.end()) << "Unknown vocabulary word: " << word;
  return it->second;
}

}  // namespace deepmath
