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

#include "hol/general.h"

namespace hol {

bool type_match(const TypePtr& general_type, const TypePtr& particular_type,
                std::map<TypeVar, TypePtr>* sofar) {
  if (!general_type || !particular_type) return false;
  if (general_type->is_vartype()) {
    auto it = sofar->find(general_type->dest_vartype());
    if (it != sofar->end()) return it->second == particular_type;
    (*sofar)[general_type->dest_vartype()] = particular_type;
    return true;
  }
  if (particular_type->is_vartype() ||
      std::get<0>(general_type->dest_type()) !=
          std::get<0>(particular_type->dest_type()))
    return false;
  const std::vector<TypePtr>& gen_vec = std::get<1>(general_type->dest_type());
  const std::vector<TypePtr>& par_vec =
      std::get<1>(particular_type->dest_type());
  if (gen_vec.size() != par_vec.size()) return false;
  for (uint64_t i = 0; i < gen_vec.size(); ++i) {
    if (!type_match(gen_vec[i], par_vec[i], sofar)) return false;
  }
  return true;
}

}  // namespace hol
