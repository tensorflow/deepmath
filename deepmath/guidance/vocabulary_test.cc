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

// Tests for vocabulary.h

#include "deepmath/guidance/vocabulary.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace deepmath {
namespace {

using tensorflow::Env;

class VocabTest : public testing::Test {
 public:
  VocabTest()
      : path_(tensorflow::io::JoinPath(tensorflow::testing::TmpDir(),
                                       "vocab")) {
    // We leave off the trailing newline to test that case,
    // and also include cases with whitespace before and after.
    const auto vocab = "7\n X\nX49 \n Yx \nf\ng";
    TF_CHECK_OK(WriteStringToFile(Env::Default(), path_, vocab));
  }

  const string& path() const { return path_; }

  void Check(const Vocabulary& vocab, const string& word, const int32 id) {
    EXPECT_EQ(vocab.Word(word), id);
    EXPECT_EQ(vocab.IdToWord(id), word);
  }

  void CheckSymbols(const Vocabulary& vocab) {
    Check(vocab, "=", Vocabulary::kEqual);
    Check(vocab, "$false", Vocabulary::kFalse);
    Check(vocab, "$true", Vocabulary::kTrue);
  }

 private:
  const string path_;
};

TEST_F(VocabTest, Normal) {
  Vocabulary vocab;
  vocab.Initialize(path());
  EXPECT_EQ(vocab.size(), 32 + 6);
  CheckSymbols(vocab);
  Check(vocab, "7", 32 + 0);
  Check(vocab, "X", 32 + 1);
  Check(vocab, "X49", 32 + 2);
  Check(vocab, "Yx", 32 + 3);
  Check(vocab, "f", 32 + 4);
  Check(vocab, "g", 32 + 5);
}

TEST_F(VocabTest, OneVariable) {
  Vocabulary vocab;
  vocab.Initialize(path() + ":one_variable");
  EXPECT_EQ(vocab.size(), 32 + 4);
  CheckSymbols(vocab);
  Check(vocab, "7", 32 + 0);
  Check(vocab, "X", 32 + 1);
  EXPECT_EQ(vocab.Word("Yx"), 32 + 1);  // Same as X
  EXPECT_EQ(vocab.Word("X49"), 32 + 1);  // Same as X
  Check(vocab, "f", 32 + 2);
  Check(vocab, "g", 32 + 3);
  EXPECT_EQ(vocab.Word("Z"), 32 + 1);  // Unseen variable
}

TEST_F(VocabTest, MapUnknown) {
  Vocabulary vocab;
  vocab.Initialize(path() + ":map_unknown");
  EXPECT_EQ(vocab.size(), 32 + 6);
  CheckSymbols(vocab);
  Check(vocab, "7", 32 + 0);
  Check(vocab, "X", 32 + 1);
  EXPECT_EQ(vocab.Word("X49"), 32 + 2);
  EXPECT_EQ(vocab.Word("Yx"), 32 + 3);
  Check(vocab, "f", 32 + 4);
  Check(vocab, "g", 32 + 5);
  EXPECT_EQ(vocab.Word("Z"), 32 + 2);  // unseen variable
  EXPECT_EQ(vocab.Word("z"), 32 + 2);  // Unseen token
}

TEST_F(VocabTest, MapUnknownOneVariable) {
  Vocabulary vocab;
  vocab.Initialize(path() + ":map_unknown,one_variable");
  EXPECT_EQ(vocab.size(), 32 + 4);
  CheckSymbols(vocab);
  Check(vocab, "7", 32 + 0);
  Check(vocab, "X", 32 + 1);
  EXPECT_EQ(vocab.Word("Yx"), 32 + 1);
  EXPECT_EQ(vocab.Word("X49"), 32 + 1);
  Check(vocab, "f", 32 + 2);
  Check(vocab, "g", 32 + 3);
  EXPECT_EQ(vocab.Word("Z"), 32 + 1);  // unseen variable
  EXPECT_EQ(vocab.Word("z"), 32 + 1);  // unseen token
}

}  // namespace
}  // namespace deepmath

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
