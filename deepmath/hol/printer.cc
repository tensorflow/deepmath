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

#include "deepmath/hol/general.h"
#include "deepmath/hol/kernel.h"
#include "deepmath/hol/printer.h"

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

static void print_training_type(std::ostream& out, const TypePtr& ty) {
  if (ty->is_vartype()) {
    out << " VT";
  } else {  // ty->is_type()
    TypeCon i = std::get<0>(ty->dest_type());
    std::vector<TypePtr> l = std::get<1>(ty->dest_type());
    out << " t" << i;
    if (!l.empty()) {
      for (uint64_t i = 0; i < l.size(); ++i) {
        print_training_type(out, l[i]);
      }
    }
  }
}

void print_training_tokens_parenthesize(std::ostream& out,
                                        const TermPtr& term) {
  if (term->is_const()) {
    out << "c" << std::get<0>(term->dest_const());
    print_training_type(out, std::get<1>(term->dest_const()));
  } else if (term->is_var()) {
    out << "v";
    print_training_type(out, std::get<1>(term->dest_var()));
  } else if (term->is_comb()) {
    TermPtr terml, termr;
    tie(terml, termr) = term->dest_comb();
    out << "(";
    print_training_tokens_parenthesize(out, terml);
    out << " ";
    print_training_tokens_parenthesize(out, termr);
    out << ")";
  } else {  // term->is_abs()
    TermVar var;
    TypePtr type;
    TermPtr subterm;
    tie(var, type, subterm) = term->dest_abs();
    out << "<v";
    print_training_type(out, type);
    out << ".";
    print_training_tokens_parenthesize(out, subterm);
    out << ">";
  }
}

const ConstId const_forall = 5;

static void print_training_tokens_vars(
    std::map<std::tuple<TermVar, TypePtr>, uint64_t,
    VariableOrdering>* bound_vars,
    std::map<std::tuple<TermVar, TypePtr>, uint64_t,
    VariableOrdering>* free_vars,
    std::ostream& out, const TermPtr& term, bool types) {
  if (term->is_const()) {
    if (arity(term->type_of()) > 0) out << " part";
    out << " c" << get_const_syntax(std::get<0>(term->dest_const()));
    if (types) print_training_type(out, term->type_of());
  } else if (term->is_var()) {
    auto it = bound_vars->find(term->dest_var());
    if (it != bound_vars->end()) {
      out << " b" << it->second;
    } else {
      auto it2 = free_vars->find(term->dest_var());
      if (it2 != free_vars->end()) {
        out << " f" << it2->second;
        if (types) print_training_type(out, term->type_of());
      } else {
        const auto ret = free_vars->size();
        out << " f" << ret;
        (*free_vars)[term->dest_var()] = ret;
      }
    }
  } else if (term->is_comb()) {
    if (term->rator()->is_const() &&
        std::get<0>(term->rator()->dest_const()) == const_forall &&
        term->rand()->is_abs()) {
      TermVar var;
      TypePtr type;
      TermPtr subterm;
      tie(var, type, subterm) = term->rand()->dest_abs();
      out << " !";
      if (types) print_training_type(out, type);
      auto v = std::make_tuple(var, type);
      int32_t erased = -1;
      auto it = bound_vars->find(v);
      if (it != bound_vars->end()) erased = it->second;
      (*bound_vars)[v] = bound_vars->size();
      print_training_tokens_vars(bound_vars, free_vars, out, subterm, types);
      if (erased != -1)
        (*bound_vars)[v] = erased;
      else
        bound_vars->erase(v);
    } else {
      TermPtr hop;
      std::vector<TermPtr> args;
      tie(hop, args) = strip_comb(term);
      if (hop->is_const() && arity(hop->type_of()) == args.size()) {
        out << " c" << get_const_syntax(std::get<0>(hop->dest_const()));
        if (types) print_training_type(out, hop->type_of());
        for (const auto& arg : args)
          print_training_tokens_vars(bound_vars, free_vars, out, arg, types);
      } else {
        TermPtr terml, termr;
        tie(terml, termr) = term->dest_comb();
        out << " *";
        print_training_tokens_vars(bound_vars, free_vars, out, terml, types);
        print_training_tokens_vars(bound_vars, free_vars, out, termr, types);
      }
    }
  } else {  // term->is_abs()
    TermVar var;
    TypePtr type;
    TermPtr subterm;
    tie(var, type, subterm) = term->dest_abs();
    out << " /";
    if (types) print_training_type(out, term->type_of());
    auto v = std::make_tuple(var, type);
    int32_t erased = -1;
    auto it = bound_vars->find(v);
    if (it != bound_vars->end()) erased = it->second;
    (*bound_vars)[v] = bound_vars->size();
    print_training_tokens_vars(bound_vars, free_vars, out, subterm, types);
    if (erased != -1)
      (*bound_vars)[v] = erased;
    else
      bound_vars->erase(v);
  }
}

void print_training_tokens(std::ostream& out, const TermPtr& term, bool types) {
  std::map<std::tuple<TermVar, TypePtr>, uint64_t, VariableOrdering>
      bound_vars{};
  std::map<std::tuple<TermVar, TypePtr>, uint64_t, VariableOrdering>
      free_vars{};
  print_training_tokens_vars(&bound_vars, &free_vars, out, strip_forall(term),
                             types);
}

}  // namespace hol
