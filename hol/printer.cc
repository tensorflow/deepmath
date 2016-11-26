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

#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "hol/kernel.h"
#include "hol/printer.h"

#include "tensorflow/core/platform/logging.h"

namespace hol {

std::unordered_map<TypeCon, std::string> type_syntax{{type_con_bool, "bool"},
  {type_con_fun, "->"}};

void declare_type_syntax(TypeCon ty_con, const std::string& syntax) {
  CHECK(type_syntax.find(ty_con) == type_syntax.end());
  type_syntax[ty_con] = syntax;
}

std::ostream& operator<<(std::ostream& out, const TypePtr& ty) {
  if (ty == nullptr) {
    out << "null";
  } else if (ty->is_vartype()) {
    out << static_cast<std::string>(ty->dest_vartype());
  } else {  // ty->is_type()
    TypeCon i = std::get<0>(ty->dest_type());
    std::vector<TypePtr> l = std::get<1>(ty->dest_type());
    if (i == type_con_fun) {
      out << "(" << l[0] << "->" << l[1] << ")";
    } else {
      auto it = type_syntax.find(i);
      CHECK(it != type_syntax.end());
      out << it->second;
      if (!l.empty()) {
        out << "(";
        for (uint64_t i = 0; i < l.size(); ++i) {
          out << l[i];
          if (i + 1 < l.size()) out << ",";
        }
        out << ")";
      }
    }
  }
  return out;
}

std::unordered_map<ConstId, std::string> const_syntax{{const_eq, "="}};

void declare_const_syntax(ConstId cnst, const std::string& syntax) {
  CHECK(const_syntax.find(cnst) == const_syntax.end());
  const_syntax[cnst] = syntax;
}

const std::string& get_const_syntax(ConstId cnst) {
  auto it = const_syntax.find(cnst);
  CHECK(it != const_syntax.end());
  return it->second;
}

bool is_infix(ConstId const_id) {
  const std::unordered_set<std::string> infix_constants = {
      "=",        "/\\",      "==>",      "\\/",   "o",         ",",
      "+",        "*",        "EXP",      "<=",    "<",         ">=",
      ">",        "-",        "DIV",      "MOD",   "treal_add", "treal_mul",
      "treal_le", "treal_eq", "+",        "*",     "<=",        "-",
      "<",        ">=",       ">",        "pow",   "/",         "<=",
      "<",        ">=",       ">",        "+",     "-",         "*",
      "pow",      "div",      "rem",      "==",    "divides",   "divides",
      "IN",       "INSERT",   "UNION",    "INTER", "DIFF",      "DELETE",
      "SUBSET",   "PSUBSET",  "HAS_SIZE", "CROSS", "<=_c",      "<_c",
      "=_c",      ">=_c",     ">_c",      "..",    "$",         "PCROSS"};
  const auto& name = get_const_syntax(const_id);
  return infix_constants.find(name) != infix_constants.end();
}

bool is_binder(ConstId const_id) {
  const std::unordered_set<std::string> binder_constants = {
      "!", "?", "?!", "@", "minimal", "lambda"};
  const auto& name = get_const_syntax(const_id);
  return binder_constants.find(name) != binder_constants.end();
}

std::ostream& operator<<(std::ostream& out, const TermPtr& term) {
  if (term == nullptr) {
    out << "null";
  } else if (term->is_const()) {
    out << get_const_syntax(std::get<0>(term->dest_const()));
  } else if (term->is_var()) {
    out << static_cast<std::string>(std::get<0>(term->dest_var()));
  } else if (term->is_comb()) {
    if (term->rator()->is_comb() && term->rator()->rator()->is_const() &&
        is_infix(std::get<0>(term->rator()->rator()->dest_const()))) {
      out << "(" << term->rator()->rand() << " "
          << get_const_syntax(std::get<0>(term->rator()->rator()->dest_const()))
          << " " << term->rand() << ")";
    } else if (term->rator()->is_const() && term->rand()->is_abs() &&
               is_binder(std::get<0>(term->rator()->dest_const()))) {
      TermVar var;
      TypePtr type;
      TermPtr subterm;
      tie(var, type, subterm) = term->rand()->dest_abs();
      out << "(" << get_const_syntax(std::get<0>(term->rator()->dest_const()))
          << var << ". " << subterm << ")";
    } else {
      TermPtr terml, termr;
      tie(terml, termr) = term->dest_comb();
      out << "(" << terml << " " << termr << ")";
    }
  } else {  // term->is_abs()
    TermVar var;
    TypePtr type;
    TermPtr subterm;
    tie(var, type, subterm) = term->dest_abs();
    out << "(\\" << static_cast<std::string>(var) << ". " << subterm << ")";
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const ThmPtr& thm) {
  if (!thm) {
    out << "null";
  } else {
    for (auto it = thm->hyps_.begin(); it != thm->hyps_.end();) {
      out << *it;
      ++it;
      if (it != thm->hyps_.end()) out << ", ";
    }
    out << " |- " << thm->concl_;
  }
  return out;
}

}  // namespace hol
