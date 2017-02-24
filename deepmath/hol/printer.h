/* Copyright 2016 Google Inc. All Rights Reserved.

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

#ifndef DEEPMATH_HOL_PRINTER_H_
#define DEEPMATH_HOL_PRINTER_H_

#include "deepmath/hol/kernel.h"

namespace hol {

std::ostream& operator<<(std::ostream& out, const TypePtr& ty);
std::ostream& operator<<(std::ostream& out, const TermPtr& term);
std::ostream& operator<<(std::ostream& out, const ThmPtr& thm);

void print_training_tokens(std::ostream& out, const TermPtr& term, bool types);

// Declares print syntax for most recently declared string
void declare_type_syntax(TypeCon, const std::string&);
void declare_const_syntax(ConstId, const std::string&);

}  // namespace hol

#endif  // DEEPMATH_HOL_PRINTER_H_
