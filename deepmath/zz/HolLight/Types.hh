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

#ifndef ZZ__HolLight__Types_hh
#define ZZ__HolLight__Types_hh

#include "Hashing.hh"
#include "List.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Atomic types: (strings mapped to integer IDs 1, 2, 3...)


struct Cnst_   {};
struct Var_    {};
struct TCon_   {};
struct TVar_   {};
struct Axiom_  {};
struct TThm_   {};

typedef Atomic<Cnst_>  Cnst;
typedef Atomic<Var_>   Var;         // NOTE! Variables must not start with a back-tick "`".
typedef Atomic<TCon_>  TCon;
typedef Atomic<TVar_>  TVar;
typedef Atomic<Axiom_> Axiom;
typedef Atomic<TThm_>  TThm;

Make_IdBase_MkIndex(Cnst);
Make_IdBase_MkIndex(Var);
Make_IdBase_MkIndex(TCon);
Make_IdBase_MkIndex(TVar);
Make_IdBase_MkIndex(Axiom);
Make_IdBase_MkIndex(TThm);


// Predefined atoms (for efficiency):
struct Type;
extern TCon tcon_bool;    // = TCon("bool")
extern TCon tcon_fun;     // = TCon("fun")
extern TCon tcon_ind;     // = TCon("ind")
extern Type type_bool;    // = Type(tcon_bool, nil)
extern Type type_alpha;   // = Type(TVar("A"))
extern Type type_booleq;  // = (the type "bool->bool->bool")
extern Cnst cnst_eq;      // = Cnst("=")
extern Cnst cnst_equiv;   // = Cnst("<=>")
extern Cnst cnst_hilb;    // = Cnst("@")
extern Cnst cnst_lam;     // = Cnst("\\")
extern Cnst cnst_iand;    // = Cnst("`&")
extern Cnst cnst_NUMERAL; // = Cnst("NUMERAL");
extern Cnst cnst_BIT0;    // = Cnst("BIT0");
extern Cnst cnst_BIT1;    // = Cnst("BIT1");
extern Cnst cnst__0;      // = Cnst("_0");


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Type = tvar(TVar) | tapp(TCon,List<Type>)


struct TypeData {
    id_t data[2];
};
template<> struct Hash_default<TypeData> : Hash_mem<TypeData> {};


struct Type : Composite<TypeData> {
    using P = Composite<TypeData>;     // -- parent type
    using P::P;
    Type() : P() {}

#if !defined(OPAQUE_HOL_TYPES)
    bool is_tvar() const { return me().data[1] == id_NULL; }
    bool is_tapp() const { return !is_tvar(); }

    // tvar : TVar
    explicit Type(TVar v) : P({TypeData{{+v, 0}}}) {}
    TVar tvar() const { assert(is_tvar()); return TVar(me().data[0]); }

    // tapp : (TCon,List<Type>)
    explicit Type(TCon c, List<Type> ts) : P({TypeData{{+ts, +c}}}) {}      // -- low-level; consider using 'kernel_Inst_Type'
    TCon       tcon () const { assert(is_tapp()); return TCon(me().data[1]); }
    List<Type> targs() const { return List<Type>(me().data[0]); }
        // -- Example: for (List<Type> it = t.targs(); it; ++it) { <code with *it> }
#endif
};

Make_IdBase_MkIndex(Type);


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Term = var(Var) | cnst(Cnst) | comb(Term,Term) | abs(Var,Type,Term)


struct Subst;   // -- defined below 'Term'
struct TSubst;


struct TermData {
    uint var_mask;      // -- abstraction of the set of unbound variables in subterm
    uint lambda_c : 30; // -- lambda counter: next free variable for lambda abstraction (named `0, `1, `2...)
    uint kind_    : 2;
    id_t type;
    id_t data[2];
};
template<> struct Hash_default<TermData> : Hash_mem<TermData> {};


// Term type. NOTE! Don't use constructors directly, use factory functions below.
struct Term : Composite<TermData> {
    using P = Composite<TermData>;     // -- parent type
    using P::P;
    Term() : P() {}

#if !defined(OPAQUE_HOL_TYPES)
    struct Tag_var {};     // -- tag-types for constructors; they are so many we opt for factory functions (see below)
    struct Tag_cnst{};
    struct Tag_comb{};
    struct Tag_abs {};
    struct Tag_unsafe_abs {};
    struct Tag_unsafe_var {};

  //________________________________________
  //  Constructors/selectors:

    enum Kind { VAR, CNST, COMB, ABS };
    Kind kind() const { return Kind(me().kind_); }
    bool is_composite() const { return kind() >= 2; }

    bool is_var () const { return kind() == VAR ; }
    bool is_cnst() const { return kind() == CNST; }
    bool is_comb() const { return kind() == COMB; }
    bool is_abs () const { return kind() == ABS ; }

    Type type() const { return Type(me().type); }

    // var : (var :Var, type :Type)
    explicit Term(Tag_var, Var x, Type ty) : P({TermData{varMask(x), 0, VAR, +ty, {+x, 0}}}) { assert(!isLambdaVar(x)); }
    Var  var  () const { assert(is_var()); return Var(me().data[0]); }

    // cnst : (cnst :Cnst, type :Type)
    explicit Term(Tag_cnst, Cnst c, Type ty) : P({TermData{0, 0, CNST, +ty, {+c, 0}}}) {}
    Cnst cnst () const { assert(is_cnst()); return Cnst(me().data[0]); }

    // comb : (fun :Term, arg :Term) -- function-application `(fun arg)`
    explicit Term(Tag_comb, Term f, Term t) : P({TermData{(f->var_mask | t->var_mask), max_(f->lambda_c, t->lambda_c), COMB, +appType(f, t), {+f, +t}}}) {}
    Term fun() const { assert(is_comb()); return Term(me().data[0]); }  // -- "rator"
    Term arg() const { assert(is_comb()); return Term(me().data[1]); }  // -- "rand"

    // abs : (avar :Term, aterm :Term) -- lambda-abstraction `\avar . aterm`
    explicit Term(Tag_abs, Term x, Term tm) : P(mkAbsData(x, tm)) {}
    Term avar () const { assert(is_abs()); return Term(me().data[0]); }     // }- low-level; consider using 'betaRed()'
    Term aterm() const { assert(is_abs()); return Term(me().data[1]); }     // }  (these methods may leak internal lambda-vars)
        // -- NOTE! Abstraction is the only non-trivial constructor. It will normalize terms to use De Bruijn level.

  //________________________________________
  //  Public methods:

    Term varSubst (Vec<Subst>& subs);
    Term typeSubst(Vec<TSubst>& subs);

    bool mayHave(Term x) const { assert(x.is_var()); return varMask(x.var()) & me().var_mask; }    // -- if FALSE, variable `x` is definitely not in this term
    bool hasVar(Term x) const;
    Term betaRed(Term x) const { assert(is_abs()); assert(x.is_var()); assert(!isLambdaVar(x.var())); return subst(aterm(), avar(), x); }   // -- `x` is term variable

    static bool isLambdaVar(Var v) { return Str(v)[0] == '`'; }

  //________________________________________
  //  Private Helpers:

    explicit Term(Tag_unsafe_abs, Term x, Term tm, Type ty_result) : P({tm->var_mask, tm->lambda_c + 1u, ABS, +ty_result,      {+x, +tm}}) {}
    explicit Term(Tag_unsafe_abs, Term x, Term tm)                 : P({tm->var_mask, tm->lambda_c + 1u, ABS, +absType(x, tm), {+x, +tm}}) {}
    explicit Term(Tag_unsafe_var, Var x, Type ty) : P({varMask(x), 0, VAR, +ty, {+x, 0}}) {}    // -- no lambda-var check

    static uint varMask(Var v) { return 1u << (+v & 31); }
    static Var  lambdaVar(uint idx);    // <<== change return type to 'Term' (accepting type as second argument -- only use of unsafeVar)

    static Term subst(Term tm, Term x, Term new_x);  // -- low-level substitution, may break lambda-invariant
    static TermData mkAbsData(Term x, Term tm);

    static Type absType(Term x, Term tm) { return Type(tcon_fun, mkList({x.type(), tm.type()})); }
    static Type appType(Term f, Term t) {   // -- check types `f : A->B`, `t : A` and return `B`
        Type ty = f.type(); assert(ty.is_tapp()); assert(ty.tcon() == tcon_fun); assert(t.type() == ty.targs()[0]);
        return ty.targs()[1]; }

    static Type tsubstType(Type ty, Vec<TSubst> const& subs);
    static Term tsubstTerm(Term tm, Vec<TSubst> const& subs);
    static Term vsubst    (Term tm, Vec<Subst> const& subs);
    static Term vsubst_ref(Term tm, Vec<Subst> const& subs);    // -- for debugging
#endif
};


#if !defined(OPAQUE_HOL_TYPES)
// FACTORY FUNCTIONS: (tm = term)
inline Term tmVar (Var  x, Type ty) { return Term(Term::Tag_var (), x, ty); }
inline Term tmCnst(Cnst c, Type ty) { return Term(Term::Tag_cnst(), c, ty); }   // -- low-level; consider using 'kernel_Inst_Cnst'
inline Term tmComb(Term f, Term tm) { return Term(Term::Tag_comb(), f, tm); }
inline Term tmAbs (Term x, Term tm) { return Term(Term::Tag_abs (), x, tm); }
#endif


// Represents substitution "x := tm".
struct Subst {
    Term x;     // -- var term
    Term tm;
    Subst(Term x = Term(), Term tm = Term()) : x(x), tm(tm) {}
};


// Represents type-substitution "a := ty".
struct TSubst {
    Type a;     // -- tvar type ('a' for 'alpha', common notation for type variables)
    Type ty;
    TSubst(Type a = Type(), Type ty = Type()) : a(a), ty(ty) {}
};

Make_IdBase_MkIndex(Term);


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Thm  = (hyps=Set<Term>, concl=Term)


struct ThmData {
    id_t data[2];
    id_t proof;
};
template<> struct Hash_default<ThmData> : Hash_mem<ThmData> {};


struct Thm : Composite<ThmData> {
    using P = Composite<ThmData>;     // -- parent type
    using P::P;
    Thm() : P() {}

#if !defined(OPAQUE_HOL_TYPES)
    explicit Thm(Term concl, List<IdBase> proof = id_NULL)                  : P({ThmData{{id_NULL, +concl}, +proof}}) {}
    explicit Thm(SSet<Term> hyps, Term concl, List<IdBase> proof = id_NULL) : P({ThmData{{+hyps,   +concl}, +proof}}) {}
    SSet<Term>   hyps () const { return SSet<Term>(me().data[0]); }
    Term         concl() const { return Term(me().data[1]); }
    List<IdBase> proof() const { return List<IdBase>(me().proof); }
#endif
};

Make_IdBase_MkIndex(Thm);


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif
