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

#include "hol/general.h"
#include "hol/kernel.h"
#include "hol/printer.h"
#include "hol/trace.h"

namespace hol {

std::vector<TypePtr> types;
std::vector<TermPtr> terms;
std::vector<ThmPtr> thms;
std::map<std::string, TypeCon> name_type_map;
std::map<std::string, TypeVar> name_typevar_map;
std::map<std::string, ConstId> name_const_map;
std::map<std::string, TermVar> name_var_map;
std::map<uint64_t, std::tuple<ThmPtr, ThmPtr> > type_defs;

TypePtr get_type(std::istringstream& iss) {
  std::string part;
  iss >> part;
  int64_t i = stoi(part);
  TypePtr ret = types[std::abs(i)];
  if (i < 0) types[std::abs(i)] = nullptr;
  return ret;
}

TermPtr get_term(std::istringstream& iss) {
  std::string part;
  iss >> part;
  int64_t i = stoi(part);
  TermPtr ret = terms[std::abs(i)];
  if (i < 0) terms[std::abs(i)] = nullptr;
  return ret;
}

ThmPtr get_thm(std::istringstream& iss) {
  std::string part;
  iss >> part;
  int64_t i = stoi(part);
  ThmPtr ret = thms[std::abs(i)];
  if (i < 0) thms[std::abs(i)] = nullptr;
  return ret;
}

void read_trace(const std::string& fname, bool print_proved, bool debug) {
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
  std::string line, rest, part;
  while (getline(ic, line)) {
    if (debug) std::cout << line << std::endl;
    std::istringstream iss(line.substr(1));
    if (line[0] == 'a') {
      iss >> part;
      if (name_type_map.find(part) == name_type_map.end()) {
        std::cout << "Unknown type: " << part << std::endl;
        break;
      }
      TypeCon type_con = name_type_map[part];
      std::vector<TypePtr> args;
      for (uint64_t i = 0; i < get_type_arity(type_con); ++i)
        args.push_back(get_type(iss));
      auto ty = mk_type(type_con, args);
      if (debug)
        std::cout << "ty (a) " << types.size() << "\t" << ty << std::endl;
      types.push_back(ty);
    } else if (line[0] == 't') {
      iss >> part;
      if (name_typevar_map.find(part) == name_typevar_map.end())
        name_typevar_map[part] = part;
      TypeVar type_var = name_typevar_map[part];
      auto ty = mk_vartype(type_var);
      if (debug)
        std::cout << "ty (t) " << types.size() << "\t" << ty << std::endl;
      types.push_back(ty);
    } else if (line[0] == 'c') {
      iss >> part;
      if (name_const_map.find(part) == name_const_map.end()) {
        std::cout << "Unknown const: " << part << std::endl;
        break;
      }
      ConstId const_id = name_const_map[part];
      TypePtr type = get_type(iss);
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
    } else if (line[0] == 'v') {
      iss >> part;
      if (name_var_map.find(part) == name_var_map.end())
        name_var_map[part] = part;
      TermVar term_var = name_var_map[part];
      TypePtr type = get_type(iss);
      auto tm = mk_var(term_var, type);
      if (debug)
        std::cout << "tm (v) " << terms.size() << "\t" << tm << " : " << type
                  << std::endl;
      terms.push_back(tm);
    } else if (line[0] == 'l') {
      TermPtr var = get_term(iss);
      TermPtr subterm = get_term(iss);
      TermPtr tm = nullptr;
      if (var) tm = mk_abs(var->dest_var(), subterm);
      if (debug)
        std::cout << "tm (l) " << terms.size() << "\t" << tm << " : "
                  << tm->type_of() << std::endl;
      terms.push_back(tm);
    } else if (line[0] == 'f') {
      TermPtr terml = get_term(iss);
      TermPtr termr = get_term(iss);
      auto tm = mk_comb(terml, termr);
      if (debug)
        std::cout << "tm (f) " << terms.size() << "\t" << tm << " : "
                  << tm->type_of() << std::endl;
      terms.push_back(tm);
    } else if (line[0] == 'F') {
      std::string const_name;
      iss >> const_name;
      TermPtr term = get_term(iss);
      ThmPtr thm = new_basic_definition(term);
      auto const_id = std::get<0>(thm->concl_->rator()->rand()->dest_const());
      declare_const_syntax(const_id, const_name);
      if (debug)
        std::cout << " (F) " << thms.size() << "\t" << thm << std::endl;
      thms.push_back(thm);
      name_const_map[const_name] = const_id;
    } else if (line[0] == 'R') {
      TermPtr term = get_term(iss);
      ThmPtr thm = REFL(term);
      if (debug)
        std::cout << " (R) " << thms.size() << "\t" << thm << std::endl;
      thms.push_back(thm);
    } else if (line[0] == 'C') {
      ThmPtr thml = get_thm(iss);
      ThmPtr thmr = get_thm(iss);
      ThmPtr thm = MK_COMB(thml, thmr);
      if (debug)
        std::cout << " (C) " << thms.size() << "\t" << thm << std::endl;
      thms.push_back(thm);
    } else if (line[0] == 'E') {
      ThmPtr thml = get_thm(iss);
      ThmPtr thmr = get_thm(iss);
      ThmPtr thm = EQ_MP(thml, thmr);
      if (debug)
        std::cout << " (E) " << thms.size() << "\t" << thm << std::endl;
      thms.push_back(thm);
    } else if (line[0] == 'H') {
      TermPtr term = get_term(iss);
      ThmPtr thm = ASSUME(term);
      if (debug)
        std::cout << " (H) " << thms.size() << "\t" << thm << std::endl;
      thms.push_back(thm);
    } else if (line[0] == 'D') {
      ThmPtr thml = get_thm(iss);
      ThmPtr thmr = get_thm(iss);
      ThmPtr thm = DEDUCT_ANTISYM(thml, thmr);
      if (debug)
        std::cout << " (D) " << thms.size() << "\t" << thm << std::endl;
      thms.push_back(thm);
    } else if (line[0] == 'B') {
      TermPtr term = get_term(iss);
      ThmPtr thm = BETA(term);
      if (debug)
        std::cout << " (B) " << thms.size() << "\t" << thm << std::endl;
      thms.push_back(thm);
    } else if (line[0] == 'S') {
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
    } else if (line[0] == 'Q') {
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
    } else if (line[0] == 'T') {
      ThmPtr thml = get_thm(iss);
      ThmPtr thmr = get_thm(iss);
      ThmPtr thm = TRANS(thml, thmr);
      if (debug)
        std::cout << " (T) " << thms.size() << "\t" << thm << std::endl;
      thms.push_back(thm);
    } else if (line[0] == 'L') {
      TermPtr term = get_term(iss);
      ThmPtr thm = nullptr;
      if (term) thm = ABS(term->dest_var(), get_thm(iss));
      if (debug)
        std::cout << " (L) " << thms.size() << "\t" << thm << std::endl;
      thms.push_back(thm);
    } else if (line[0] == 'A') {
      iss >> part;
      TermPtr term = get_term(iss);
      ThmPtr thm = new_axiom(term);
      if (debug)
        std::cout << " (A) " << thms.size() << "\t" << thm << std::endl;
      thms.push_back(thm);
    } else if (line[0] == 'Y') {
      std::string type_name, abs_name, rep_name;
      iss >> type_name;
      iss >> abs_name;
      iss >> rep_name;
      iss >> part;
      iss >> part;
      ThmPtr thm = get_thm(iss);
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
    } else if (line[0] == '1') {
      iss >> part;
      ThmPtr thm = std::get<0>(type_defs[std::abs(stoi(part))]);
      if (debug)
        std::cout << " (1) " << thms.size() << "\t" << thm << std::endl;
      thms.push_back(thm);
    } else if (line[0] == '2') {
      iss >> part;
      ThmPtr thm = std::get<1>(type_defs[std::abs(stoi(part))]);
      if (debug)
        std::cout << " (2) " << thms.size() << "\t" << thm << std::endl;
      thms.push_back(thm);
    } else if (line[0] == '+') {
      if (print_proved && line[1] != '!') {  // Human named theorem
        iss >> part;
        std::cout << part << " " << thms[thms.size() - 1] << std::endl;
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
    if (!debug && !print_proved && thms.size() % 10000 == 0)
      std::cout << "." << std::flush;
  }
}

}  // namespace hol
