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

#ifndef DEEPMATH_HOL_GENERAL_H_
#define DEEPMATH_HOL_GENERAL_H_

#include "deepmath/hol/kernel.h"

namespace hol {

bool type_match(const TypePtr& general_type, const TypePtr& particular_type,
                std::map<TypeVar, TypePtr>* sofar);

uint64_t arity(TypePtr type);

std::tuple<TermPtr, std::vector<TermPtr> > strip_comb(TermPtr term);

TermPtr strip_forall(TermPtr term);

}  // namespace hol

#endif  // DEEPMATH_HOL_GENERAL_H_
