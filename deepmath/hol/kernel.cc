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

// Kernel of a higher-order logic prover,
// heavily inspired by the one of HOL Light:
// https://github.com/jrh13/hol-light/blob/master/fusion.ml

#include <algorithm>

#include "deepmath/hol/kernel.h"

namespace hol {

bool Type::operator==(const Type& rhs) const {
  if (this == &rhs)
    return true;
  else if (is_vartype())
    return (rhs.is_vartype() && dest_vartype() == rhs.dest_vartype());
  else if (!rhs.is_type() ||
           std::get<0>(dest_type()) != std::get<0>(rhs.dest_type()))
    return false;
  const auto& lhs_args = std::get<1>(dest_type());
  const auto& rhs_args = std::get<1>(rhs.dest_type());
  for (uint64_t i = 0; i < lhs_args.size(); ++i)
    if (*lhs_args[i] != *rhs_args[i]) return false;
  return true;
}

bool Type::operator<(const Type& rhs) const {
  if (this == &rhs)
    return false;
  else if (is_vartype())
    return (rhs.is_type() || dest_vartype() < rhs.dest_vartype());
  else if (!rhs.is_type())
    return false;
  if (std::get<0>(dest_type()) < std::get<0>(rhs.dest_type())) return true;
  if (std::get<0>(dest_type()) > std::get<0>(rhs.dest_type())) return false;
  const auto& lhs_args = std::get<1>(dest_type());
  const auto& rhs_args = std::get<1>(rhs.dest_type());
  for (uint64_t i = 0; i < lhs_args.size(); ++i) {
    if (*(lhs_args[i]) < *(rhs_args[i])) return true;
    if (*(rhs_args[i]) < *(lhs_args[i])) return false;
  }
  return false;
}

Type::~Type() {
  if (is_vartype())
    type_var_.~TypeVar();
  if (is_type())
    type_.~tuple();
}

// The arities of the currently declared types. The types
// "bool" with arity 0 and "fun" with arity 2 are used already
// in the inference rules so they are predefined.

static std::vector<uint64_t> the_type_constants{0, 2};

// The types of the currently declared term constants.
// Polymorphic equality is predefined.

// TODO(geoffreyi): Deal with thread safety.
static std::vector<TypePtr> the_term_constants{mk_type(
    type_con_fun,
    std::vector<TypePtr>{
        type_alpha,
        mk_type(type_con_fun, std::vector<TypePtr>{type_alpha, type_bool})})};

uint64_t get_type_arity(TypeCon type_con) {
  return the_type_constants[type_con];
}

TypeCon new_type(uint64_t arity) {
  the_type_constants.push_back(arity);
  return the_type_constants.size() - 1;
}

TypePtr mk_vartype(const TypeVar type_var) {
  return std::make_shared<Type>(type_var, Type::Secret());
}

TypePtr mk_type(const TypeCon type_con, const std::vector<TypePtr>& args) {
  if ((type_con >= the_type_constants.size() ||
       the_type_constants[type_con] != args.size()) &&
      type_con > type_con_fun)
    return nullptr;
  for (const auto& arg : args)
    if (!arg) return nullptr;
  return Type::unsafe_tyapp(type_con, args);
}

std::set<TypeVar> tyvars(const TypePtr& ty) {
  std::set<TypeVar> ret;
  if (ty->is_vartype()) {
    ret.insert(ty->dest_vartype());
  } else {  // ty->is_type()
    for (const TypePtr& t : std::get<1>(ty->dest_type()))
      for (const auto& tv : tyvars(t)) ret.insert(tv);
  }
  return ret;
}

TypePtr type_subst(const std::map<TypeVar, TypePtr>& instantiation,
                   const TypePtr& type) {
  for (const auto& pair : instantiation)
    if (!pair.second) return nullptr;
  if (type->is_vartype()) {
    const TypeVar type_var = type->dest_vartype();
    auto it = instantiation.find(type_var);
    if (it == instantiation.end()) return type;
    return it->second;
  } else {
    TypeCon type_con;
    std::vector<TypePtr> args;
    tie(type_con, args) = type->dest_type();
    std::vector<TypePtr> new_args;
    for (const auto& arg : args) {
      TypePtr subtype = type_subst(instantiation, arg);
      if (!subtype) return nullptr;
      new_args.push_back(subtype);
    }
    return Type::unsafe_tyapp(type_con, new_args);
  }
}

bool VariableOrdering::operator()(const std::tuple<TermVar, TypePtr>& a,
                                  const std::tuple<TermVar, TypePtr>& b) const {
  if (std::get<0>(a) < std::get<0>(b)) return true;
  if (std::get<0>(a) > std::get<0>(b)) return false;
  return (*std::get<1>(a) < *std::get<1>(b));
}

bool Term::operator==(const Term& rhs) const {
  if (this == &rhs) return true;
  if (is_var())
    return rhs.is_var() &&
            std::get<0>(dest_var()) == std::get<0>(rhs.dest_var()) &&
            *(std::get<1>(dest_var())) == *(std::get<1>(rhs.dest_var()));
  if (is_const())
    return rhs.is_const() &&
            std::get<0>(dest_const()) == std::get<0>(rhs.dest_const()) &&
            *(std::get<1>(dest_const())) == *(std::get<1>(rhs.dest_const()));
  if (is_comb())
    return rhs.is_comb() &&
            *(std::get<0>(dest_comb())) == *(std::get<0>(rhs.dest_comb())) &&
            *(std::get<1>(dest_comb())) == *(std::get<1>(rhs.dest_comb()));
  return rhs.is_abs() &&
          std::get<0>(dest_abs()) == std::get<0>(rhs.dest_abs()) &&
          *(std::get<1>(dest_abs())) == *(std::get<1>(rhs.dest_abs())) &&
          *(std::get<2>(dest_abs())) == *(std::get<2>(rhs.dest_abs()));
}

TypePtr Term::type_of() const {
  if (is_var()) {
    return std::get<1>(dest_var());
  } else if (is_const()) {
    return std::get<1>(dest_const());
  } else if (is_comb()) {
    const TermPtr& l = std::get<0>(dest_comb());
    const TypePtr& lt = l->type_of();
    return (std::get<1>(lt->dest_type()))[1];
  } else {  // is_abs()
    TermVar var;
    TypePtr type;
    TermPtr subterm;
    tie(var, type, subterm) = dest_abs();
    return Type::unsafe_tyapp(type_con_fun,
                              std::vector<TypePtr>{type, subterm->type_of()});
  }
}

Term::~Term() {
  if (is_var())
    term_var_.~tuple();
  else if (is_const())
    term_const_.~tuple();
  else if (is_comb())
    term_comb_.~tuple();
  else  // is_abs()
    term_abs_.~tuple();
}

TermPtr mk_var(TermVar var_id, const TypePtr& type) {
  if (!type) return nullptr;
  return Term::unsafe_var(var_id, type);
}

TermPtr mk_var(const std::tuple<TermVar, TypePtr>& value) {
  if (!std::get<1>(value)) return nullptr;
  return Term::unsafe_var(value);
}

ConstId new_constant(const TypePtr& type) {
  the_term_constants.push_back(type);
  return static_cast<ConstId>(the_term_constants.size() - 1);
}

const TypePtr& get_const_type(ConstId const_id) {
  return the_term_constants[const_id];
}

TermPtr mk_const(ConstId const_id, const std::map<TypeVar, TypePtr>& subst) {
  if (const_id < the_term_constants.size())
    return Term::unsafe_const(const_id,
                              type_subst(subst, the_term_constants[const_id]));
  else
    return nullptr;
}

TermPtr mk_abs(TermVar term_var, const TypePtr& type, const TermPtr& term) {
  if (!type || !term)
    return nullptr;
  else
    return Term::unsafe_abs(term_var, type, term);
}

TermPtr mk_abs(const std::tuple<TermVar, TypePtr>& var, const TermPtr& term) {
  return mk_abs(std::get<0>(var), std::get<1>(var), term);
}

TermPtr mk_comb(const TermPtr& function, const TermPtr& argument) {
  if (!function || !argument) return nullptr;
  const TypePtr& function_type = function->type_of();
  if (!function_type->is_type() ||
      std::get<0>(function_type->dest_type()) != type_con_fun ||
      *((std::get<1>(function_type->dest_type()))[0]) != *argument->type_of())
    return nullptr;
  return Term::unsafe_comb(function, argument);
}

TermPtr Term::unsafe_mk_eq(const TermPtr& lhs, const TermPtr& rhs) {
  const auto& lhs_ty = lhs->type_of();
  const auto& eq = Term::unsafe_const(
      const_eq, Type::unsafe_tyapp(
                    type_con_fun,
                    std::vector<TypePtr>{
                        lhs_ty, Type::unsafe_tyapp(
                                    type_con_fun,
                                    std::vector<TypePtr>{lhs_ty, type_bool})}));
  return Term::unsafe_comb(Term::unsafe_comb(eq, lhs), rhs);
}

// Comparator for variables modulo alpha-equality in a given variable context.
int8_t ordav(const std::vector<std::tuple<std::tuple<TermVar, TypePtr>,
                                          std::tuple<TermVar, TypePtr>>>& env,
             const std::tuple<TermVar, TypePtr>& x1,
             const std::tuple<TermVar, TypePtr>& x2) {
  if (env.empty() && x1 == x2) return 0;
  for (auto p = env.rbegin(); p != env.rend(); ++p) {
    std::tuple<TermVar, TypePtr> t1, t2;
    tie(t1, t2) = *p;
    if (std::get<0>(x1) == std::get<0>(t1) &&
        *std::get<1>(x1) == *std::get<1>(t1))
      return (std::get<0>(x2) == std::get<0>(t2) &&
              *std::get<1>(x2) == *std::get<1>(t2))
                 ? 0
                 : -1;
    if (std::get<0>(x2) == std::get<0>(t2) &&
        *std::get<1>(x2) == *std::get<1>(t2))
      return 1;
  }
  if (std::get<0>(x1) < std::get<0>(x2))
    return -1;
  else if (std::get<0>(x1) > std::get<0>(x2))
    return 1;
  else if (*std::get<1>(x1) < *std::get<1>(x2))
    return -1;
  else if (*std::get<1>(x2) < *std::get<1>(x1))
    return 1;
  else
    return 0;
}

// Comparator of terms modulo alpha-equality. As abstractions are encountered,
// they are stored in an environment vector, to compare free variables
// differently from bound ones.
int8_t orda(std::vector<std::tuple<std::tuple<TermVar, TypePtr>,
                                   std::tuple<TermVar, TypePtr>>>* env,
            const TermPtr& tm1, const TermPtr& tm2) {
  if (env->empty() && tm1 == tm2) {
    return 0;
  } else if (tm1->is_var() && tm2->is_var()) {
    return ordav(*env, tm1->dest_var(), tm2->dest_var());
  } else if (tm1->is_const() && tm2->is_const()) {
    if (std::get<0>(tm1->dest_const()) < std::get<0>(tm2->dest_const()))
      return -1;
    else if (std::get<0>(tm2->dest_const()) < std::get<0>(tm1->dest_const()))
      return 1;
    else if (*std::get<1>(tm1->dest_const()) < *std::get<1>(tm2->dest_const()))
      return -1;
    else if (*std::get<1>(tm2->dest_const()) < *std::get<1>(tm1->dest_const()))
      return 1;
    else
      return 0;
  } else if (tm1->is_comb() && tm2->is_comb()) {
    TermPtr l1, r1, l2, r2;
    tie(l1, r1) = tm1->dest_comb();
    tie(l2, r2) = tm2->dest_comb();
    auto c = orda(env, l1, l2);
    if (c != 0)
      return c;
    else
      return orda(env, r1, r2);
  } else if (tm1->is_abs() && tm2->is_abs()) {
    TermVar v1, v2;
    TypePtr ty1, ty2;
    TermPtr t1, t2;
    tie(v1, ty1, t1) = tm1->dest_abs();
    tie(v2, ty2, t2) = tm2->dest_abs();
    if (*ty1 < *ty2) return -1;
    if (*ty2 < *ty1) return 1;
    env->emplace_back(make_tuple(make_tuple(v1, ty1), make_tuple(v2, ty2)));
    auto ret = orda(env, t1, t2);
    env->pop_back();
    return ret;
  } else if (tm1->is_const()) {
    return -1;
  } else if (tm2->is_const()) {
    return 1;
  } else if (tm1->is_var()) {
    return -1;
  } else if (tm2->is_var()) {
    return 1;
  } else if (tm1->is_comb()) {
    return -1;
  } else {  // tm2->is_comb()
    return 1;
  }
}

int8_t alphaorder(const TermPtr& tm1, const TermPtr& tm2) {
  std::vector<
      std::tuple<std::tuple<TermVar, TypePtr>, std::tuple<TermVar, TypePtr>>>
      env;
  return orda(&env, tm1, tm2);
}

bool TermOrdering::operator()(const TermPtr& a, const TermPtr& b) const {
  return alphaorder(a, b) == -1;
}

// TODO(geoffreyi): Use an auxiliary function to avoid repeatedly making
// small sets?
std::set<std::tuple<TermVar, TypePtr>> Term::frees() const {
  std::set<std::tuple<TermVar, TypePtr>> ret;
  if (is_var()) {
    ret.insert(dest_var());
  } else if (is_comb()) {
    TermPtr l, r;
    tie(l, r) = dest_comb();
    ret = l->frees();
    for (const auto& t : r->frees()) ret.insert(t);
  } else if (is_abs()) {
    TermVar var;
    TypePtr type;
    TermPtr subterm;
    tie(var, type, subterm) = dest_abs();
    ret = subterm->frees();
    ret.erase(make_tuple(var, type));
  }
  return ret;
}

bool Term::freesin(
    std::set<std::tuple<TermVar, TypePtr>, VariableOrdering>* var_set) const {
  if (is_var()) {
    return var_set->find(dest_var()) != var_set->end();
  } else if (is_const()) {
    return true;
  } else if (is_comb()) {
    TermPtr l, r;
    tie(l, r) = dest_comb();
    return l->freesin(var_set) && r->freesin(var_set);
  } else {  // term->is_abs()
    TermVar var_id;
    TypePtr type;
    TermPtr term;
    tie(var_id, type, term) = dest_abs();
    const auto var = make_tuple(var_id, type);
    if (var_set->find(var) != var_set->end()) return term->freesin(var_set);
    // The modification to var_set is always undone.
    var_set->insert(var);
    auto ret = term->freesin(var_set);
    var_set->erase(var);
    return ret;
  }
}

bool vfree_in(const std::tuple<TermVar, TypePtr>& vv, const TermPtr& term) {
  if (term->is_abs()) {
    TermVar v;
    TypePtr ty;
    TermPtr t;
    tie(v, ty, t) = term->dest_abs();
    return (v != std::get<0>(vv) || *ty != *std::get<1>(vv)) && vfree_in(vv, t);
  } else if (term->is_comb()) {
    TermPtr l, r;
    tie(l, r) = term->dest_comb();
    return vfree_in(vv, l) || vfree_in(vv, r);
  } else if (!term->is_var()) {
    return false;
  } else {  // term->is_abs()
    return std::get<0>(term->dest_var()) == std::get<0>(vv) &&
           *std::get<1>(term->dest_var()) == *std::get<1>(vv);
  }
}

template <typename T>
std::set<T> set_union(const std::set<T>& s1, const std::set<T>& s2) {
  std::set<T> res;
  std::set_union(s1.begin(), s1.end(), s2.begin(), s2.end(),
                 inserter(res, res.begin()));
  return res;
}

template <typename T, typename O>
std::set<T, O> set_union(const std::set<T, O>& s1, const std::set<T, O>& s2) {
  std::set<T, O> res;
  std::set_union(s1.begin(), s1.end(), s2.begin(), s2.end(),
                 inserter(res, res.begin()));
  return res;
}

std::set<TypeVar> Term::type_vars_in_term() const {
  if (is_var()) return tyvars(std::get<1>(dest_var()));
  if (is_const()) return tyvars(std::get<1>(dest_const()));
  if (is_comb()) {
    TermPtr l, r;
    tie(l, r) = dest_comb();
    return set_union(l->type_vars_in_term(), r->type_vars_in_term());
  }
  TermVar v;
  TypePtr ty;
  TermPtr t;
  tie(v, ty, t) = dest_abs();
  return set_union(tyvars(ty), t->type_vars_in_term());
}

// Renames the given variable to avoid all free variable names
// from the given set.
std::tuple<TermVar, TypePtr> variant(const std::set<TermPtr>& avoid,
                                     const std::tuple<TermVar, TypePtr>& v) {
  for (const auto& t : avoid)
    if (vfree_in(v, t)) {
      std::string vn;
      TypePtr vt;
      tie(vn, vt) = v;
      std::string vn2 = vn + "'";
      return variant(avoid, make_tuple(static_cast<TermVar>(vn2), vt));
    }
  return v;
}

// Recursive part of variable substitution, assumes that arguments
// are correct.
TermPtr Term::vsubst_rec(Substitution* subst, const TermPtr& term) {
  if (term->is_var()) {
    const auto& it = subst->find(term->dest_var());
    if (it == subst->end())
      return term;
    else
      return it->second;
  } else if (term->is_const()) {
    return term;
  } else if (term->is_comb()) {
    const TermPtr& terml = term->rator();
    const TermPtr& termr = term->rand();
    TermPtr terml2 = vsubst_rec(subst, terml);
    TermPtr termr2 = vsubst_rec(subst, termr);
    if (terml == terml2 && termr == termr2)
      return term;
    else
      return unsafe_comb(terml2, termr2);
  } else {  // term->is_abs()
    TermVar var_id = std::get<0>(term->dest_abs());
    const TypePtr& type = std::get<1>(term->dest_abs());
    const TermPtr& subterm = std::get<2>(term->dest_abs());
    const auto var = make_tuple(var_id, type);
    const auto& it = subst->find(var);
    TermPtr erased;
    if (it != subst->end()) {
      erased = it->second;
      subst->erase(it);
    }
    if (subst->empty()) {
      if (erased != nullptr) (*subst)[var] = erased;
      return term;
    }
    TermPtr subterm2 = vsubst_rec(subst, subterm);
    if (*subterm == *subterm2) {
      if (erased != nullptr) (*subst)[var] = erased;
      return term;
    }
    for (const auto& subst_pair : *subst) {
      if (vfree_in(var, subst_pair.second) &&
          vfree_in(subst_pair.first, subterm)) {
        auto subst2 = *subst;
        if (erased != nullptr) (*subst)[var] = erased;
        auto var2 = variant(std::set<TermPtr>{subterm2}, var);
        subst2[var] = Term::unsafe_var(var2);
        auto var2_id = std::get<0>(var2);
        return Term::unsafe_abs(var2_id, type, vsubst_rec(&subst2, subterm));
      }
    }
    if (erased != nullptr) (*subst)[var] = erased;
    return unsafe_abs(var_id, type, subterm2);
  }
}

// Variable substitution. Checks if the arguments are correct and
// calls vsubst_rec
TermPtr vsubst(Substitution* subst, const TermPtr& term) {
  if (subst->empty() || !term) return term;
  for (const auto& subst_pair : *subst)
    if (std::get<1>(subst_pair.first) == nullptr ||
        subst_pair.second == nullptr ||
        *std::get<1>(subst_pair.first) != *subst_pair.second->type_of())
      return nullptr;
  return Term::vsubst_rec(subst, term);
}

// Instantiation of a type may require variable renaming, for example
// (\x:A x:B. P (x:A) (x:B))[A->B] requires renaming the appropriate binder.
std::tuple<TermPtr, bool> Term::inst_rec(
    const std::map<TermPtr, TermPtr, TermOrdering>& env,
    const std::map<TypeVar, TypePtr>& instantiation, const TermPtr& term) {
  if (term->is_var()) {
    const TermVar term_var = std::get<0>(term->dest_var());
    const TypePtr& type = std::get<1>(term->dest_var());
    const TypePtr& type2 = type_subst(instantiation, type);
    const auto var2 = make_tuple(term_var, type2);
    const TermPtr& term2 = (*type == *type2) ? term : Term::unsafe_var(var2);
    const auto& it = env.find(term2);
    if (it == env.end() || *it->second == *term)
      return make_tuple(term2, true);
    else
      return make_tuple(term2, false);
  } else if (term->is_const()) {
    const ConstId const_id = std::get<0>(term->dest_const());
    const TypePtr& type = std::get<1>(term->dest_const());
    const TypePtr& type2 = type_subst(instantiation, type);
    if (*type == *type2)
      return make_tuple(term, true);
    else
      return make_tuple(Term::unsafe_const(const_id, type2), true);
  } else if (term->is_comb()) {
    const TermPtr& terml0 = std::get<0>(term->dest_comb());
    const TermPtr& termr0 = std::get<1>(term->dest_comb());
    const auto& terml = inst_rec(env, instantiation, terml0);
    if (!std::get<1>(terml)) return terml;
    const auto& termr = inst_rec(env, instantiation, termr0);
    if (!std::get<1>(termr)) return termr;
    if (std::get<0>(terml) == terml0 && std::get<0>(termr) == termr0)
      return make_tuple(term, true);
    else
      return make_tuple(
          Term::unsafe_comb(std::get<0>(terml), std::get<0>(termr)), true);
  } else {  // term->is_abs()
    TermVar var = std::get<0>(term->dest_abs());
    const TypePtr& type = std::get<1>(term->dest_abs());
    const TermPtr& subterm = std::get<2>(term->dest_abs());
    const TermPtr var_y = Term::unsafe_var(var, type);
    const TermPtr var_y2 = std::get<0>(inst_rec(
        std::map<TermPtr, TermPtr, TermOrdering>(), instantiation, var_y));
    std::map<TermPtr, TermPtr, TermOrdering> env2 = env;
    env2[var_y2] = var_y;
    auto subterm2 = inst_rec(env2, instantiation, subterm);
    if (std::get<1>(subterm2)) {  // No clash
      if (*var_y2 == *var_y && *std::get<0>(subterm2) == *subterm)
        return make_tuple(term, true);
      else
        return make_tuple(Term::unsafe_abs(std::get<0>(var_y2->dest_var()),
                                           std::get<1>(var_y2->dest_var()),
                                           std::get<0>(subterm2)),
                          true);
    }
    if (*std::get<0>(subterm2) != *var_y2) {  // Clash with a different variable
      return subterm2;
    }
    // Clash with the current binder
    std::set<TermPtr> ifrees;
    for (const auto& free_var : subterm->frees())
      ifrees.insert(
          std::get<0>(inst_rec(std::map<TermPtr, TermPtr, TermOrdering>(),
                               instantiation, Term::unsafe_var(free_var))));
    auto var_y3 = variant(ifrees, var_y2->dest_var());
    auto var_z = Term::unsafe_var(std::get<0>(var_y3), type);
    Substitution subst{{var_y->dest_var(), var_z}};
    auto term3 = Term::unsafe_abs(std::get<0>(var_z->dest_var()),
                                  std::get<1>(var_z->dest_var()),
                                  vsubst(&subst, subterm));
    return inst_rec(env, instantiation, term3);
  }
}

TermPtr inst(const std::map<TypeVar, TypePtr>& instantiation,
             const TermPtr& term) {
  if (instantiation.empty() || !term) return term;
  for (const auto& inst_pair : instantiation)
    if (!inst_pair.second) return nullptr;
  return std::get<0>(Term::inst_rec(std::map<TermPtr, TermPtr, TermOrdering>(),
                                    instantiation, term));
}

ThmPtr REFL(const TermPtr& term) {
  if (!term) return nullptr;
  return Thm::mk(std::set<TermPtr, TermOrdering>{},
                 Term::unsafe_mk_eq(term, term));
}

bool is_eq(const TermPtr& term) {
  if (!term || !term->is_comb() || !term->rator()->is_comb()) return false;
  const TermPtr& termll = term->rator()->rator();
  return termll->is_const() && std::get<0>(termll->dest_const()) == const_eq;
}

ThmPtr TRANS(const ThmPtr& left_eq, const ThmPtr& right_eq) {
  if (!left_eq || !right_eq || !is_eq(left_eq->concl_) ||
      !is_eq(right_eq->concl_))
    return nullptr;
  const TermPtr& left_eq_lhs = left_eq->concl_->rator();
  const TermPtr& left_rhs = left_eq->concl_->rand();
  const TermPtr& right_rhs = right_eq->concl_->rand();
  const TermPtr& right_lhs = right_eq->concl_->rator()->rand();
  if (alphaorder(left_rhs, right_lhs) != 0)
    return nullptr;
  else
    return Thm::mk(set_union(left_eq->hyps_, right_eq->hyps_),
                   Term::unsafe_comb(left_eq_lhs, right_rhs));
}

ThmPtr MK_COMB(const ThmPtr& fun_eq, const ThmPtr& arg_eq) {
  if (!fun_eq || !arg_eq || !is_eq(fun_eq->concl_) || !is_eq(arg_eq->concl_))
    return nullptr;
  const TermPtr& fun_eql = std::get<0>(fun_eq->concl_->dest_comb());
  const TermPtr& right_fun = std::get<1>(fun_eq->concl_->dest_comb());
  const TermPtr& arg_eql = std::get<0>(arg_eq->concl_->dest_comb());
  const TermPtr& right_arg = std::get<1>(arg_eq->concl_->dest_comb());
  const TermPtr& left_fun = std::get<1>(fun_eql->dest_comb());
  const TermPtr& left_arg = std::get<1>(arg_eql->dest_comb());
  const TypePtr& right_type = right_fun->type_of();
  if (!right_type->is_type()) return nullptr;
  TypeCon type_con = std::get<0>(right_type->dest_type());
  const std::vector<TypePtr>& type_args = std::get<1>(right_type->dest_type());
  if (type_con != type_con_fun || *(type_args[0]) != *(right_arg->type_of()))
    return nullptr;
  else
    return Thm::mk(set_union(fun_eq->hyps_, arg_eq->hyps_),
                   Term::unsafe_mk_eq(Term::unsafe_comb(left_fun, left_arg),
                                      Term::unsafe_comb(right_fun, right_arg)));
}

ThmPtr ABS(const std::tuple<TermVar, TypePtr>& var, const ThmPtr& eq) {
  if (!std::get<1>(var) || !eq || !is_eq(eq->concl_)) return nullptr;
  for (const auto& hyp : eq->hyps_)
    if (vfree_in(var, hyp)) return nullptr;
  TermPtr eq_lhs, rhs;
  tie(eq_lhs, rhs) = eq->concl_->dest_comb();
  const TermPtr lhs = std::get<1>(eq_lhs->dest_comb());
  return Thm::mk(eq->hyps_,
                 Term::unsafe_mk_eq(mk_abs(var, lhs),
                                    Term::unsafe_abs(std::get<0>(var),
                                                     std::get<1>(var), rhs)));
}

ThmPtr BETA(const TermPtr& term) {
  if (!term || !term->is_comb()) return nullptr;
  TermPtr terml, termr;
  tie(terml, termr) = term->dest_comb();
  if (!terml->is_abs()) return nullptr;
  TermVar var;
  TypePtr type;
  TermPtr subterm;
  tie(var, type, subterm) = terml->dest_abs();
  if (!termr->is_var() || var != std::get<0>(termr->dest_var()) ||
      *type != *(std::get<1>(termr->dest_var())))
    return nullptr;
  return Thm::mk(std::set<TermPtr, TermOrdering>{},
                 Term::unsafe_mk_eq(term, subterm));
}

ThmPtr ASSUME(const TermPtr& term) {
  if (!term) return nullptr;
  const TypePtr& type = term->type_of();
  if (!type->is_type() || std::get<0>(type->dest_type()) != type_con_bool)
    return nullptr;
  else
    return Thm::mk(std::set<TermPtr, TermOrdering>{term}, term);
}

ThmPtr EQ_MP(const ThmPtr& prop_eq, const ThmPtr& prop) {
  if (!prop_eq || !prop || !is_eq(prop_eq->concl_)) return nullptr;
  TermPtr prop_eq_left, prop_right;
  tie(prop_eq_left, prop_right) = prop_eq->concl_->dest_comb();
  const TermPtr& prop_left = std::get<1>(prop_eq_left->dest_comb());
  if (alphaorder(prop_left, prop->concl_) != 0) return nullptr;
  return Thm::mk(set_union(prop_eq->hyps_, prop->hyps_), prop_right);
}

ThmPtr DEDUCT_ANTISYM(const ThmPtr& thm1, const ThmPtr& thm2) {
  if (!thm1 || !thm2) return nullptr;
  std::set<TermPtr, TermOrdering> hyps;
  // The conclusions usually but not always are among hyps.
  for (const auto& hyp1 : thm1->hyps_)
    if (alphaorder(hyp1, thm2->concl_) != 0) hyps.insert(hyp1);
  for (const auto& hyp2 : thm2->hyps_)
    if (alphaorder(hyp2, thm1->concl_) != 0) hyps.insert(hyp2);
  // In 91% of the cases hyps.size <= 1, 4%=2, 3%=3
  return Thm::mk(hyps, Term::unsafe_mk_eq(thm1->concl_, thm2->concl_));
}

ThmPtr INST_TYPE(const std::map<TypeVar, TypePtr>& instantiation,
                 const ThmPtr& thm) {
  if (!thm) return nullptr;
  std::set<TermPtr, TermOrdering> new_hyps;
  for (const auto& hyp : thm->hyps_) new_hyps.insert(inst(instantiation, hyp));
  TermPtr new_concl = inst(instantiation, thm->concl_);
  if (!new_concl) return nullptr;
  return Thm::mk(new_hyps, new_concl);
}

ThmPtr INST(Substitution* subst, const ThmPtr& thm) {
  if (!thm) return nullptr;
  std::set<TermPtr, TermOrdering> new_hyps;
  for (const auto& hyp : thm->hyps_) new_hyps.insert(vsubst(subst, hyp));
  TermPtr new_concl = vsubst(subst, thm->concl_);
  if (!new_concl) return nullptr;
  return Thm::mk(new_hyps, new_concl);
}

static std::vector<ThmPtr> the_axioms = {};

ThmPtr new_axiom(const TermPtr& term) {
  if (!term) return nullptr;
  const TypePtr& type = term->type_of();
  if (!type->is_type() || std::get<0>(type->dest_type()) != type_con_bool)
    return nullptr;
  ThmPtr ret = Thm::mk(std::set<TermPtr, TermOrdering>{}, term);
  the_axioms.push_back(ret);
  return ret;
}

static std::vector<ThmPtr> the_definitions = {};

ThmPtr new_basic_definition(const TermPtr& definition) {
  std::set<std::tuple<TermVar, TypePtr>, VariableOrdering> empty_set{};
  if (!definition || !definition->freesin(&empty_set)) return nullptr;
  TypePtr constant_type = definition->type_of();
  std::set<TypeVar> constant_type_vars = tyvars(constant_type);
  for (const auto& tyvar : definition->type_vars_in_term())
    if (constant_type_vars.find(tyvar) == constant_type_vars.end())
      return nullptr;
  ConstId const_id = new_constant(constant_type);
  TermPtr const_lhs = Term::unsafe_const(const_id, constant_type);
  ThmPtr ret = Thm::mk(std::set<TermPtr, TermOrdering>{},
                       Term::unsafe_mk_eq(const_lhs, definition));
  the_definitions.push_back(ret);
  return ret;
}

std::tuple<TypeCon, std::tuple<ConstId, ConstId>, std::tuple<ThmPtr, ThmPtr>>
new_basic_type_definition(const ThmPtr& existence_proof) {
  const auto error = std::make_tuple(0, std::make_tuple(0, 0),
                                     std::make_tuple(nullptr, nullptr));
  if (!existence_proof || !existence_proof->hyps_.empty() ||
      !existence_proof->concl_->is_comb())
    return error;
  TermPtr property, witness;
  tie(property, witness) = existence_proof->concl_->dest_comb();
  std::set<std::tuple<TermVar, TypePtr>, VariableOrdering> empty_set{};
  if (!property->freesin(&empty_set)) return error;
  std::set<TypeVar> tyvars = property->type_vars_in_term();
  TypeCon type_name = static_cast<TypeCon>(the_type_constants.size());
  the_type_constants.push_back(tyvars.size());
  std::vector<TypePtr> type_args;
  for (const auto& tyvar : tyvars) type_args.push_back(mk_vartype(tyvar));
  TypePtr new_type = mk_type(type_name, type_args);
  TypePtr old_type = witness->type_of();
  TypePtr abs_type = Type::unsafe_tyapp(
      type_con_fun, std::vector<TypePtr>{old_type, new_type});
  TypePtr rep_type = Type::unsafe_tyapp(
      type_con_fun, std::vector<TypePtr>{new_type, old_type});
  auto abs_const_id = new_constant(abs_type);
  auto rep_const_id = new_constant(rep_type);
  TermPtr abs_const = Term::unsafe_const(abs_const_id, abs_type);
  TermPtr rep_const = Term::unsafe_const(rep_const_id, rep_type);
  TermPtr new_var = Term::unsafe_var(static_cast<TermVar>("a"), new_type);
  TermPtr old_var = Term::unsafe_var(static_cast<TermVar>("r"), old_type);
  ThmPtr ret1 = Thm::mk(
      std::set<TermPtr, TermOrdering>{},
      Term::unsafe_mk_eq(
          Term::unsafe_comb(abs_const, Term::unsafe_comb(rep_const, new_var)),
          new_var));
  ThmPtr ret2 =
      Thm::mk(std::set<TermPtr, TermOrdering>{},
              Term::unsafe_mk_eq(
                  Term::unsafe_comb(property, old_var),
                  Term::unsafe_mk_eq(
                      Term::unsafe_comb(rep_const,
                                        Term::unsafe_comb(abs_const, old_var)),
                      old_var)));
  return std::make_tuple(type_name, std::make_tuple(abs_const_id, rep_const_id),
                         std::make_tuple(ret1, ret2));
}

}  // namespace hol
