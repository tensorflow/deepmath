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

#ifndef DEEPMATH_HOL_KERNEL_H_
#define DEEPMATH_HOL_KERNEL_H_

#include <stdint.h>

#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace hol {

class Type;
class Term;
class Thm;

typedef std::shared_ptr<const Type> TypePtr;
typedef std::shared_ptr<const Term> TermPtr;
typedef std::shared_ptr<const Thm> ThmPtr;

// TODO(geoffreyi) make a Symbol class which uses fast pointer comparison
typedef std::string TypeVar;
typedef uint64_t TypeCon;
typedef std::string TermVar;
typedef uint64_t ConstId;

// This corresponds to the ML type:
//
// type hol_type = private
//   Tyvar of string
// | Tyapp of string *  hol_type list

class Type final {
  // Private struct for make_shared purposes
  struct Secret {};

 public:
  bool operator==(const Type& rhs) const;
  bool operator!=(const Type& rhs) const { return !(*this == rhs); }
  bool operator<(const Type& rhs) const;
  bool is_vartype() const { return kind_ == TYVAR; }
  bool is_type() const { return kind_ == TYAPP; }
  const TypeVar dest_vartype() const { return type_var_; }
  const std::tuple<TypeCon, std::vector<TypePtr> >& dest_type() const {
    return type_;
  }
  friend TypePtr mk_vartype(const TypeVar value);
  friend TypePtr mk_type(const TypeCon type_con,
                         const std::vector<TypePtr>& args);
  friend TypePtr type_subst(const std::map<TypeVar, TypePtr>& instantiation,
                            const TypePtr& term);
  friend std::tuple<TypeCon, std::tuple<ConstId, ConstId>,
                    std::tuple<ThmPtr, ThmPtr> >
  new_basic_type_definition(const ThmPtr& existence_proof);
  friend std::set<TypeVar> tyvars(const TypePtr& type);
  ~Type();

  // Public but uncallable so that make_shared works
  explicit Type(const TypeVar value, Secret)
      : kind_(TYVAR), type_var_(value) {}
  Type(const TypeCon type_con, const std::vector<TypePtr>& args, Secret)
      : kind_(TYAPP), type_(make_tuple(type_con, args)) {}

 private:
  friend class Term;

  static TypePtr unsafe_tyapp(const TypeCon type_con,
                              const std::vector<TypePtr>& args) {
    return std::make_shared<Type>(type_con, args, Secret());
  }

  enum TypeKind { TYVAR, TYAPP };
  // kind_ remembers if the type is a type variable or type application
  const TypeKind kind_;
  union {
    const TypeVar type_var_;
    const std::tuple<TypeCon, std::vector<TypePtr> > type_;
  };
};

TypePtr mk_vartype(const TypeVar type_var);

TypeCon new_type(uint64_t arity);
uint64_t get_type_arity(TypeCon type_con);

class VariableOrdering {
 public:
  bool operator()(const std::tuple<TermVar, TypePtr>& a,
                  const std::tuple<TermVar, TypePtr>& b) const;
};

typedef std::map<std::tuple<TermVar, TypePtr>, TermPtr, VariableOrdering>
    Substitution;

class TermOrdering {
 public:
  bool operator()(const TermPtr& a, const TermPtr& b) const;
};

// This corresponds to the ML type:
//
// type term = private
//   Var of string * hol_type
// | Const of string * hol_type
// | Comb of term * term
// | Abs of term * term

class Term final {
  // Private struct for make_shared purposes
  struct Secret {};

 public:
  bool operator==(const Term& rhs) const;
  bool operator!=(const Term& rhs) const { return !(*this == rhs); }
  bool is_var() const { return kind_ == VAR; }
  bool is_const() const { return kind_ == CONST; }
  bool is_comb() const { return kind_ == COMB; }
  bool is_abs() const { return kind_ == ABS; }
  TypePtr type_of() const;
  // Returns the free variables of a term
  std::set<std::tuple<TermVar, TypePtr> > frees() const;
  // Checks if all free variables of a term appear in set
  bool freesin(std::set<std::tuple<TermVar, TypePtr>, VariableOrdering>*) const;
  // Returns the free type variables of a term
  std::set<TypeVar> type_vars_in_term() const;
  const std::tuple<TermVar, TypePtr>& dest_var() const { return term_var_; }
  const std::tuple<ConstId, TypePtr>& dest_const() const { return term_const_; }
  const std::tuple<TermPtr, TermPtr>& dest_comb() const { return term_comb_; }
  const std::tuple<TermVar, TypePtr, TermPtr>& dest_abs() const {
    return term_abs_;
  }
  // Return the operator and the operand of an application respectively
  const TermPtr& rator() const { return std::get<0>(dest_comb()); }
  // "rand()" is the name used in HOL that we want to keep even if
  // NOLINTNEXTLINE "lint" believes it is a call to random number generation
  const TermPtr& rand() const { return std::get<1>(dest_comb()); }
  friend TermPtr mk_var(const TermVar term_var, const TypePtr& type);
  friend TermPtr mk_var(const std::tuple<TermVar, TypePtr>& value);
  friend TermPtr mk_const(ConstId const_id,
                          const std::map<TypeVar, TypePtr>& subst);
  friend TermPtr mk_comb(const TermPtr& terml, const TermPtr& termr);
  friend TermPtr mk_abs(TermVar term_var, const TypePtr& type,
                        const TermPtr& term);
  friend TermPtr mk_abs(const std::tuple<TermVar, TypePtr>& var,
                        const TermPtr& term);
  // Substitution of terms for variables
  friend TermPtr vsubst(Substitution* subst, const TermPtr& term);
  // Type instantiation
  friend TermPtr inst(const std::map<TypeVar, TypePtr>& instantiation,
                      const TermPtr& term);
  friend ThmPtr REFL(const TermPtr& term);
  friend ThmPtr TRANS(const ThmPtr& left_eq, const ThmPtr& right_eq);
  friend ThmPtr MK_COMB(const ThmPtr& fun_eq, const ThmPtr& arg_eq);
  friend ThmPtr ABS(const std::tuple<TermVar, TypePtr>& var, const ThmPtr& eq);
  friend ThmPtr BETA(const TermPtr& term);
  friend ThmPtr DEDUCT_ANTISYM(const ThmPtr& thm1, const ThmPtr& thm2);

  friend ThmPtr INST(Substitution* subst, const ThmPtr& thm);
  friend ThmPtr new_basic_definition(const TermPtr& definition);
  friend std::tuple<TypeCon, std::tuple<ConstId, ConstId>,
                    std::tuple<ThmPtr, ThmPtr> >
  new_basic_type_definition(const ThmPtr& existence_proof);
  ~Term();

  // Public but uncallable for make_shared purposes
  Term(const TermVar i, const TypePtr& t, Secret)
      : kind_(VAR), term_var_(std::make_tuple(i, t)) {}
  explicit Term(const std::tuple<TermVar, TypePtr>& value, Secret)
      : kind_(VAR), term_var_(value) {}
  Term(const ConstId i, const TypePtr& t, Secret)
      : kind_(CONST), term_const_(std::make_tuple(i, t)) {}
  Term(const TermPtr& l, const TermPtr& r, Secret)
      : kind_(COMB), term_comb_(std::make_tuple(l, r)) {}
  Term(const TermVar i, const TypePtr& ty, const TermPtr& t, Secret)
      : kind_(ABS), term_abs_(std::make_tuple(i, ty, t)) {}

 private:
  static TermPtr unsafe_var(const TermVar term_var, const TypePtr& type) {
    return std::make_shared<Term>(term_var, type, Secret());
  }
  static TermPtr unsafe_var(const std::tuple<TermVar, TypePtr>& value) {
    return std::make_shared<Term>(value, Secret());
  }
  static TermPtr unsafe_const(const ConstId const_id, const TypePtr& type) {
    return std::make_shared<Term>(const_id, type, Secret());
  }
  static TermPtr unsafe_comb(const TermPtr& l, const TermPtr& r) {
    return std::make_shared<Term>(l, r, Secret());
  }
  static TermPtr unsafe_abs(const TermVar v, const TypePtr& ty,
                            const TermPtr& t) {
    return std::make_shared<Term>(v, ty, t, Secret());
  }
  static TermPtr unsafe_mk_eq(const TermPtr& lhs, const TermPtr& rhs);
  static TermPtr vsubst_rec(Substitution* s, const TermPtr& tm);
  static std::tuple<TermPtr, bool> inst_rec(
      const std::map<TermPtr, TermPtr, TermOrdering>& env,
      const std::map<TypeVar, TypePtr>& instantiation, const TermPtr& term);
  enum TermKind { VAR, CONST, COMB, ABS };
  // kind_ stores the term constructor used to build the term
  const int8_t kind_;
  union {
    const std::tuple<TermVar, TypePtr> term_var_;
    const std::tuple<ConstId, TypePtr> term_const_;
    const std::tuple<TermPtr, TermPtr> term_comb_;
    const std::tuple<TermVar, TypePtr, TermPtr> term_abs_;
  };
};

TermPtr mk_var(TermVar var_id, const TypePtr& type);
TermPtr mk_const(ConstId const_id, const std::map<TypeVar, TypePtr>& subst);
TermPtr mk_abs(TermVar term_var, const TypePtr& type, const TermPtr& term);
TermPtr mk_comb(const TermPtr& function, const TermPtr& argument);

ConstId new_constant(const TypePtr& type);
const TypePtr& get_const_type(ConstId const_id);

// Comparator on terms, which returns 0 for alpha-equal terms
int8_t alphaorder(const TermPtr& tm1, const TermPtr& tm2);

const TypeCon type_con_bool = static_cast<TypeCon>(0);
const TypeCon type_con_fun = static_cast<TypeCon>(1);
const TypeVar type_var_alpha = static_cast<TypeVar>("A");
const ConstId const_eq = static_cast<ConstId>(0);

const TypePtr type_bool = mk_type(type_con_bool, std::vector<TypePtr>());
const TypePtr type_alpha = mk_vartype(type_var_alpha);

class Thm final {
  // Private struct for make_shared purposes
  struct Secret {};

 public:
  const std::set<TermPtr, TermOrdering> hyps_;
  const TermPtr concl_;
  friend ThmPtr REFL(const TermPtr& term);
  friend ThmPtr TRANS(const ThmPtr& left_eq, const ThmPtr& right_eq);
  friend ThmPtr MK_COMB(const ThmPtr& fun_eq, const ThmPtr& arg_eq);
  friend ThmPtr ABS(const std::tuple<TermVar, TypePtr>& var, const ThmPtr& eq);
  friend ThmPtr ASSUME(const TermPtr& term);
  friend ThmPtr BETA(const TermPtr& term);
  friend ThmPtr EQ_MP(const ThmPtr& prop_eq, const ThmPtr& prop);
  friend ThmPtr DEDUCT_ANTISYM(const ThmPtr& thm1, const ThmPtr& thm2);
  friend ThmPtr INST_TYPE(const std::map<TypeVar, TypePtr>& inst,
                          const ThmPtr& thm);
  friend ThmPtr INST(Substitution* subst, const ThmPtr& thm);
  friend ThmPtr new_axiom(const TermPtr& term);
  friend ThmPtr new_basic_definition(const TermPtr& definition);
  friend std::tuple<TypeCon, std::tuple<ConstId, ConstId>,
                    std::tuple<ThmPtr, ThmPtr> >
  new_basic_type_definition(const ThmPtr& existence_proof);

  // Public but uncallable for make_shared use
  Thm(const std::set<TermPtr, TermOrdering>& hyps, const TermPtr& concl, Secret)
      : hyps_(hyps), concl_(concl) {}

 private:
  static ThmPtr mk(const std::set<TermPtr, TermOrdering>& hyps,
                   const TermPtr& concl) {
    return std::make_shared<Thm>(hyps, concl, Secret());
  }
};

//    t
// --------
// |- t = t
ThmPtr REFL(const TermPtr& term);

// h1 |- l = m    h2 |- m' = r
// --------------------------- (provided m alpha-equal to m')
//      h1 \/ h2 |- l = r
ThmPtr TRANS(const ThmPtr& left_eq, const ThmPtr& right_eq);

// h1 |- f = g    h2 |- a = b
// -------------------------- (provided types match)
//    h1 \/ h2 |- f a = g b
ThmPtr MK_COMB(const ThmPtr& fun_eq, const ThmPtr& arg_eq);

//   v    h |- l = r
// ------------------ (provided x is free in h)
// h |- \v. l = \v. r
ThmPtr ABS(const std::tuple<TermVar, TypePtr>& var, const ThmPtr& eq);

// t : bool
// --------
//  t |- t
ThmPtr ASSUME(const TermPtr& term);

//    (\x. t) x
// ----------------
// |- (\x. t) x = t
ThmPtr BETA(const TermPtr& term);

// h1 |- f = g    h2 |- f'
// ----------------------- (if f is alpha-equal to f')
//     h1 \/ h2 |- g
ThmPtr EQ_MP(const ThmPtr& prop_eq, const ThmPtr& prop);

//      h1 |- c1    h2 |- c2
// ---------------------------------
// (h1 - c2) \/ (h2 - c1) |- c1 = c2
ThmPtr DEDUCT_ANTISYM(const ThmPtr& thm1, const ThmPtr& thm2);

//     {(v1, t1), ..., (vn, tn)}    h |- p
// --------------------------------------------
// h[v1:=t1,...,vn:=tn] |- p[v1:=t1,...,vn:=tn]
ThmPtr INST_TYPE(const std::map<TypeVar, TypePtr>& inst, const ThmPtr& thm);

//     {(x1, t1), ..., (xn, tn)}    h |- p
// --------------------------------------------
// h[x1:=t1,...,xn:=tn] |- p[x1:=t1,...,xn:=tn]
ThmPtr INST(Substitution* subst, const ThmPtr& thm);

// t : bool
// --------
//   |- t
ThmPtr new_axiom(const TermPtr& term);

//    t
// -------- (provided t has no free variables and all tyvars are in its type)
// |- c = t
ThmPtr new_basic_definition(const TermPtr& definition);

//                           |- P t
// -----------------------------------------------------------------
// ty    abs    rep    |- abs(rep a) = a   |- P r = (rep(abs r) = r)
std::tuple<TypeCon, std::tuple<ConstId, ConstId>, std::tuple<ThmPtr, ThmPtr> >
new_basic_type_definition(const ThmPtr& existence_proof);

}  // namespace hol

#endif  // DEEPMATH_HOL_KERNEL_H_
