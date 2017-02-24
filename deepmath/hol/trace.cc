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

// This program reads a trace file produced by HOL Light proofrecording and
// passes every step to the C++ kernel.

// Example invocation:
//   trace -print_proved -proofs_file proofs-nat

// The trace file in each line has a single character that denotes the step,
// followed by the arguments to the step. The steps codes are:
//  't' type variable
//  'a' type constructor
//  'v' variable
//  'c' constant
//  'f' application
//  'l' abstraction
//  'R' REFL
//  'T' TRANS
//  'C' MK_COMB
//  'L' ABS
//  'B' BETA
//  'H' ASSUME
//  'E' EQ_MP
//  'D' DEDUCT
//  'S' INST
//  'Q' INST_TYPE
//  'A' new_axiom
//  'F' new_basic_definition
//  'Y' new_basic_type_definition

#include <fstream>
#include <iostream>
#include <sstream>

#include "deepmath/hol/general.h"
#include "deepmath/hol/kernel.h"
#include "deepmath/hol/printer.h"
#include "deepmath/hol/trace.h"

namespace hol {

std::vector<TypePtr> types;
std::vector<TermPtr> terms;
std::vector<ThmPtr> thms;
std::map<std::string, TypeCon> name_type_map;
std::map<std::string, TypeVar> name_typevar_map;
std::map<std::string, ConstId> name_const_map;
std::map<std::string, TermVar> name_var_map;
std::map<uint64_t, std::tuple<ThmPtr, ThmPtr> > type_defs;

TypePtr get_type(std::ifstream& ic) {
  int64_t i;
  ic >> i;
  TypePtr ret = types[std::abs(i)];
  if (i < 0) types[std::abs(i)] = nullptr;
  return ret;
}

TermPtr get_term(std::ifstream& ic) {
  int64_t i;
  ic >> i;
  TermPtr ret = terms[std::abs(i)];
  if (i < 0) terms[std::abs(i)] = nullptr;
  return ret;
}

ThmPtr get_thm(std::ifstream& ic) {
  int64_t i;
  ic >> i;
  ThmPtr ret = thms[std::abs(i)];
  if (i < 0) thms[std::abs(i)] = nullptr;
  return ret;
}

void read_trace(const std::string& fname, bool print_proved, bool print_tokens,
                bool print_types, bool debug) {
  types.push_back(nullptr);
  terms.push_back(nullptr);
  thms.push_back(nullptr);
  name_type_map["bool"] = type_con_bool;
  name_type_map["fun"] = type_con_fun;
  auto type_con_ind = new_type(0);
  name_type_map["ind"] = type_con_ind;
  declare_type_syntax(type_con_ind, "ind");

  name_const_map["="] = const_eq;
  auto const_hilbert = new_constant(mk_type(
      type_con_fun,
      std::vector<TypePtr>{
          mk_type(type_con_fun, std::vector<TypePtr>{type_alpha, type_bool}),
          type_alpha}));
  name_const_map["@"] = const_hilbert;
  declare_const_syntax(const_hilbert, "@");

  std::ifstream ic(fname);
  char tag;
  std::string line, name, part;
  while (ic >> tag) {
    if (debug) std::cout << tag << std::endl;
    if (tag == 'a') {
      ic >> name;
      const auto it = name_type_map.find(name);
      if (it == name_type_map.end()) {
        std::cout << "Unknown type: " << name << std::endl;
        break;
      }
      TypeCon type_con = it->second;
      std::vector<TypePtr> args;
      for (uint64_t i = 0; i < get_type_arity(type_con); ++i)
        args.push_back(get_type(ic));
      auto ty = mk_type(type_con, args);
      if (debug)
        std::cout << "ty (a) " << types.size() << "\t" << ty << std::endl;
      types.push_back(ty);
    } else if (tag == 't') {
      ic >> name;
      if (name_typevar_map.find(name) == name_typevar_map.end())
        name_typevar_map[name] = name;
      TypeVar type_var = name_typevar_map[name];
      auto ty = mk_vartype(type_var);
      if (debug)
        std::cout << "ty (t) " << types.size() << "\t" << ty << std::endl;
      types.push_back(ty);
    } else if (tag == 'c') {
      ic >> name;
      if (name_const_map.find(name) == name_const_map.end()) {
        std::cout << "Unknown const: " << name << std::endl;
        break;
      }
      ConstId const_id = name_const_map[name];
      TypePtr type = get_type(ic);
      std::map<TypeVar, TypePtr> subst;
      bool matched = type_match(get_const_type(const_id), type, &subst);
      TermPtr term = nullptr;
      if (matched) {
        term = mk_const(const_id, subst);
      } else {
        std::cout << "Impossible to match constant type" << std::endl;
        exit(0);
      }
      if (debug)
        std::cout << "tm (c) " << terms.size() << "\t" << term << " : "
                  << term->type_of() << std::endl;
      terms.push_back(term);
    } else if (tag == 'v') {
      ic >> name;
      if (name_var_map.find(name) == name_var_map.end())
        name_var_map[name] = name;
      TermVar term_var = name_var_map[name];
      TypePtr type = get_type(ic);
      auto tm = mk_var(term_var, type);
      if (debug)
        std::cout << "tm (v) " << terms.size() << "\t" << tm << " : " << type
                  << std::endl;
      terms.push_back(tm);
    } else if (tag == 'l') {
      TermPtr var = get_term(ic);
      TermPtr subterm = get_term(ic);
      TermPtr tm = nullptr;
      if (var) tm = mk_abs(var->dest_var(), subterm);
      if (debug)
        std::cout << "tm (l) " << terms.size() << "\t" << tm << " : "
                  << tm->type_of() << std::endl;
      terms.push_back(tm);
    } else if (tag == 'f') {
      TermPtr terml = get_term(ic);
      TermPtr termr = get_term(ic);
      auto tm = mk_comb(terml, termr);
      if (debug)
        std::cout << "tm (f) " << terms.size() << "\t" << tm << " : "
                  << tm->type_of() << std::endl;
      terms.push_back(tm);
    } else if (tag == 'F') {
      std::string const_name;
      ic >> const_name;
      TermPtr term = get_term(ic);
      ThmPtr thm = new_basic_definition(term);
      auto const_id = std::get<0>(thm->concl_->rator()->rand()->dest_const());
      declare_const_syntax(const_id, const_name);
      if (debug)
        std::cout << " (F) " << thms.size() << "\t" << thm << std::endl;
      thms.push_back(thm);
      name_const_map[const_name] = const_id;
    } else if (tag == 'R') {
      TermPtr term = get_term(ic);
      ThmPtr thm = REFL(term);
      if (debug)
        std::cout << " (R) " << thms.size() << "\t" << thm << std::endl;
      thms.push_back(thm);
    } else if (tag == 'C') {
      ThmPtr thml = get_thm(ic);
      ThmPtr thmr = get_thm(ic);
      ThmPtr thm = MK_COMB(thml, thmr);
      if (debug)
        std::cout << " (C) " << thms.size() << "\t" << thm << std::endl;
      thms.push_back(thm);
    } else if (tag == 'E') {
      ThmPtr thml = get_thm(ic);
      ThmPtr thmr = get_thm(ic);
      ThmPtr thm = EQ_MP(thml, thmr);
      if (debug)
        std::cout << " (E) " << thms.size() << "\t" << thm << std::endl;
      thms.push_back(thm);
      if (print_tokens) {
        std::cout << "E";
        print_training_tokens(std::cout, thm->concl_, print_types);
        std::cout << std::endl;
      }
    } else if (tag == 'H') {
      TermPtr term = get_term(ic);
      ThmPtr thm = ASSUME(term);
      if (debug)
        std::cout << " (H) " << thms.size() << "\t" << thm << std::endl;
      thms.push_back(thm);
    } else if (tag == 'D') {
      ThmPtr thml = get_thm(ic);
      ThmPtr thmr = get_thm(ic);
      ThmPtr thm = DEDUCT_ANTISYM(thml, thmr);
      if (debug)
        std::cout << " (D) " << thms.size() << "\t" << thm << std::endl;
      thms.push_back(thm);
    } else if (tag == 'B') {
      TermPtr term = get_term(ic);
      ThmPtr thm = BETA(term);
      if (debug)
        std::cout << " (B) " << thms.size() << "\t" << thm << std::endl;
      thms.push_back(thm);
    } else if (tag == 'S') {
      getline(ic, line);
      std::istringstream iss(line);
      iss >> part;
      Substitution subst;
      while (!iss.eof()) {
        TermPtr var = terms[std::abs(stoi(part))];
        if (stoi(part) < 0) terms[std::abs(stoi(part))] = nullptr;
        iss >> part;
        TermPtr term = terms[std::abs(stoi(part))];
        if (stoi(part) < 0) terms[std::abs(stoi(part))] = nullptr;
        iss >> part;
        if (!var) {
          std::cout << "Variable fail!" << std::endl;
          exit(0);
        }
        if (*var != *term && subst.find(var->dest_var()) == subst.end())
          subst[var->dest_var()] = term;
      }
      ThmPtr thm = INST(&subst, thms[std::abs(stoi(part))]);
      if (stoi(part) < 0) thms[std::abs(stoi(part))] = nullptr;
      if (debug)
        std::cout << " (S) " << thms.size() << "\t" << thm << std::endl;
      thms.push_back(thm);
    } else if (tag == 'Q') {
      getline(ic, line);
      std::istringstream iss(line);
      iss >> part;
      std::map<TypeVar, TypePtr> inst;
      while (!iss.eof()) {
        TypeVar type_var = types[std::abs(stoi(part))]->dest_vartype();
        if (stoi(part) < 0) types[std::abs(stoi(part))] = nullptr;
        iss >> part;
        TypePtr type = types[std::abs(stoi(part))];
        if (stoi(part) < 0) types[std::abs(stoi(part))] = nullptr;
        iss >> part;
        if ((!type->is_vartype() || type->dest_vartype() != type_var) &&
            inst.find(type_var) == inst.end())
          inst[type_var] = type;
      }
      ThmPtr thm = INST_TYPE(inst, thms[std::abs(stoi(part))]);
      if (stoi(part) < 0) thms[std::abs(stoi(part))] = nullptr;
      if (debug)
        std::cout << " (Q) " << thms.size() << "\t" << thm << std::endl;
      thms.push_back(thm);
    } else if (tag == 'T') {
      ThmPtr thml = get_thm(ic);
      ThmPtr thmr = get_thm(ic);
      ThmPtr thm = TRANS(thml, thmr);
      if (debug)
        std::cout << " (T) " << thms.size() << "\t" << thm << std::endl;
      thms.push_back(thm);
    } else if (tag == 'L') {
      TermPtr term = get_term(ic);
      ThmPtr thm = nullptr;
      if (term) thm = ABS(term->dest_var(), get_thm(ic));
      if (debug)
        std::cout << " (L) " << thms.size() << "\t" << thm << std::endl;
      thms.push_back(thm);
    } else if (tag == 'A') {
      ic >> part;
      TermPtr term = get_term(ic);
      ThmPtr thm = new_axiom(term);
      if (debug)
        std::cout << " (A) " << thms.size() << "\t" << thm << std::endl;
      thms.push_back(thm);
    } else if (tag == 'Y') {
      std::string type_name, abs_name, rep_name;
      ic >> type_name;
      ic >> abs_name;
      ic >> rep_name;
      ic >> part;
      ic >> part;
      ThmPtr thm = get_thm(ic);
      auto defined = new_basic_type_definition(thm);
      name_type_map[type_name] = std::get<0>(defined);
      type_defs[thms.size()] = std::get<2>(defined);
      name_const_map[abs_name] = std::get<0>(std::get<1>(defined));
      name_const_map[rep_name] = std::get<1>(std::get<1>(defined));
      declare_type_syntax(std::get<0>(defined), type_name);
      declare_const_syntax(std::get<0>(std::get<1>(defined)), abs_name);
      declare_const_syntax(std::get<1>(std::get<1>(defined)), rep_name);
      if (debug)
        std::cout << " (Y) " << thms.size() << "\t" << type_name << std::endl;
      thms.push_back(nullptr);
    } else if (tag == '1') {
      ic >> part;
      ThmPtr thm = std::get<0>(type_defs[std::abs(stoi(part))]);
      if (debug)
        std::cout << " (1) " << thms.size() << "\t" << thm << std::endl;
      thms.push_back(thm);
    } else if (tag == '2') {
      ic >> part;
      ThmPtr thm = std::get<1>(type_defs[std::abs(stoi(part))]);
      if (debug)
        std::cout << " (2) " << thms.size() << "\t" << thm << std::endl;
      thms.push_back(thm);
    } else if (tag == '+') {
      getline(ic, line);
      if (print_tokens && thms[thms.size() - 1] &&
          thms[thms.size() - 1]->hyps_.empty()) {
        if (line[0] == '!') {
          std::cout << '!';
          print_training_tokens(std::cout, thms[thms.size() - 1]->concl_,
                                print_types);
          std::cout << std::endl;
        } else {
          std::cout << '+';
          print_training_tokens(std::cout, thms[thms.size() - 1]->concl_,
                                print_types);
          std::cout << std::endl;
        }
      }
      if (print_proved && line[0] != '!') {  // Human named theorem
        std::cout << line << " " << thms[thms.size() - 1] << std::endl;
      }
    } else if (tag == '-') {
      if (print_tokens && thms[thms.size() - 1] &&
          thms[thms.size() - 1]->hyps_.empty()) {
        std::cout << '-';
        print_training_tokens(std::cout, thms[thms.size() - 1]->concl_,
                              print_types);
        std::cout << std::endl;
      }
    } else {
      std::cout << line << std::endl;
      break;
    }
    if (thms.size() > 1 && thms[thms.size() - 1] == nullptr &&
        type_defs.find(thms.size() - 1) == type_defs.end()) {
      std::cout << "Incorrect theorem step " << thms.size() - 1
                << " derived from: " << line << std::endl;
      exit(0);
    }
    if (!debug && !print_proved && !print_tokens && thms.size() % 10000 == 0)
      std::cout << "." << std::flush;
  }
}

}  // namespace hol
