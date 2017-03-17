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

#include "deepmath/guidance/serialize.h"

namespace deepmath {

void SerializeTerm(SlowSerializeContext* context,
                   const deepmath::FirstOrderLogicTerm& term) {
  switch (term.term_type_case()) {
    case deepmath::FirstOrderLogicTerm::kVariable:
      context->Add(term.variable().name());
      break;
    case deepmath::FirstOrderLogicTerm::kNumber:
      context->Add(term.number().value());
      break;
    case deepmath::FirstOrderLogicTerm::kFunction: {
      context->Add(term.function().name());
      SerializeArgs(context, term.function().args());
      break;
    }
    default:
      LOG(FATAL) << "Invalid term type " << term.term_type_case();
  }
}

}  // namespace deepmath
