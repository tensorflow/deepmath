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

#include ZZ_Prelude_hh
#include "Types.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Predefined atoms (for efficiency):


TCon tcon_bool;
TCon tcon_fun;
TCon tcon_ind;
Type type_bool;
Type type_alpha;
Type type_booleq;
Cnst cnst_eq;
Cnst cnst_equiv;
Cnst cnst_hilb;
Cnst cnst_lam;
Cnst cnst_iand;
Cnst cnst_NUMERAL;
Cnst cnst_BIT0;
Cnst cnst_BIT1;
Cnst cnst__0;


ZZ_Initializer(predefined_atoms, 0) {
    tcon_bool    = TCon("bool");
    tcon_fun     = TCon("fun");
    tcon_ind     = TCon("ind");
    type_bool    = Type(tcon_bool, List<Type>());
    type_alpha   = Type(TVar("A"));
    type_booleq  = Type(tcon_fun, mkList({ type_bool, Type(tcon_fun, mkList({type_bool, type_bool})) }));
    cnst_eq      = Cnst("=");
    cnst_equiv   = Cnst("<=>");
    cnst_hilb    = Cnst("@");
    cnst_lam     = Cnst("\\");        // -- NOTE! This one is not defined in the ML kernel.
    cnst_iand    = Cnst("`&");        // -- Internal constant to represent pair of conclusions
    cnst_NUMERAL = Cnst("NUMERAL");
    cnst_BIT0    = Cnst("BIT0");
    cnst_BIT1    = Cnst("BIT1");
    cnst__0      = Cnst("_0");
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Public methods:


bool Term::hasVar(Term x) const {    // -- will return TRUE for lambda-vars under abstraction
    if (!mayHave(x)) return false;
    switch (kind()){
    case VAR : return (*this == x);
    case CNST: return false;
    case COMB: return fun().hasVar(x) || arg().hasVar(x);
    case ABS : return avar().hasVar(x) || aterm().hasVar(x);
    default: assert(false); }
}


ZZ_PTimer_Add(varSubst);
ZZ_PTimer_Add(typeSubst);


Term Term::varSubst(Vec<Subst>& subs) {
    ZZ_PTimer_Scope(varSubst);
    for (Subst s : subs){ assert(s.x.is_var()); assert(s.x.type() == s.tm.type()); }
    return vsubst(*this, subs); }


Term Term::typeSubst(Vec<TSubst>& subs) {
    ZZ_PTimer_Scope(typeSubst);
    for (TSubst s : subs) assert(s.a.is_tvar());
    return tsubstTerm(*this, subs); }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Static Helpers:


Var Term::lambdaVar(uint idx) {
    static Vec<Var> memo;
    if (!memo(idx)){
        char buf[32];
        sprintf(buf, "`%u", idx);   // -- backtick is reserved (unused symbol in the proof-logs); lambda variables are `0, `1, `2...
        //for (char* p = buf+1; *p; p++) *p += 'a' - '0';   // -- change `0 to `a, `1 to `b etc.
        memo[idx] = Var(buf);
    }
    return memo[idx];
}


// Single variable substitution (syntactic, not respecting bound variables)
Term Term::subst(Term tm, Term x, Term new_x) {
    if (!tm.mayHave(x)) return tm;
    switch (tm.kind()){
    case VAR : return (tm == x) ? new_x : tm;
    case CNST: return tm;
    case COMB: return tmComb(subst(tm.fun(), x, new_x), subst(tm.arg(), x, new_x));
    case ABS : return Term(Tag_unsafe_abs(), subst(tm.avar(), x, new_x), subst(tm.aterm(), x, new_x), tm.type());
    default: assert(false); }
}


// Create 'TermData' for 'abs', the only non-trivial constructor (due to lambda-variables).
TermData Term::mkAbsData(Term x, Term tm) {
    assert(!isLambdaVar(x.var()));
    Term alpha(Tag_unsafe_var(), lambdaVar(tm->lambda_c), x.type()); // -- create new unique lambda-variable
    tm = subst(tm, x, alpha);

    TermData d;
    d.var_mask = tm->var_mask;
    d.lambda_c = tm->lambda_c + 1u;
    d.kind_ = ABS;
    d.type = +absType(alpha, tm);
    d.data[0] = +alpha;
    d.data[1] = +tm;
    return d;
}


//=================================================================================================
// -- Type substition:


Type Term::tsubstType(Type ty, Vec<TSubst> const& subs)
{
    if (ty.is_tvar()){
        // Type variable:
        for (TSubst const& sub : subs)
            if (sub.a == ty)
                return sub.ty;
        return ty;
    }else{
        // Type constructor:
        if (ty.targs().empty()) return ty;      // -- this line could be removed (optimization)

        Vec<Type> new_targs;
        for (List<Type> it = ty.targs(); it; ++it)
            new_targs.push(tsubstType(*it, subs));
        return Type(ty.tcon(), mkList(new_targs));
    }
}


// Reference version. Can be optimized, but not a bottleneck (so keep it readable).
Term Term::tsubstTerm(Term tm, Vec<TSubst> const& subs)
{
    switch (tm.kind()){
    case Term::VAR:  return Term(Tag_unsafe_var(), tm.var(), tsubstType(tm.type(), subs));
    case Term::CNST: return tmCnst(tm.cnst(), tsubstType(tm.type(), subs));
    case Term::COMB: return tmComb(tsubstTerm(tm.fun(), subs), tsubstTerm(tm.arg(), subs));
    case Term::ABS:  return Term(Tag_unsafe_abs(), tsubstTerm(tm.avar(), subs), tsubstTerm(tm.aterm(), subs));
    default: assert(false); }
}


//=================================================================================================
// -- Variable substition:


// Reference version. More readable and used to verify the faster implementation.
Term Term::vsubst_ref(Term tm, Vec<Subst> const& subs)
{
    switch (tm.kind()){
    case Term::VAR:
        for (Subst const& sub : subs)
            if (sub.x == tm)
                return sub.tm;
        return tm;

    case Term::CNST:
        return tm;

    case Term::COMB:
        return tmComb(vsubst_ref(tm.fun(), subs), vsubst_ref(tm.arg(), subs));

    case Term::ABS: {
        char buf[32];
        sprintf(buf, " %s", tm.avar().var().c_str());     // -- add a space to the name to make it unique ('tmAbs()' will remove the variable from the final term)
        Term x = tmVar(Var(buf), tm.avar().type());
        return tmAbs(x, vsubst_ref(tm.betaRed(x), subs)); }

    default: assert(false); }
}


// Optimized implementation. Can probably do even better, but no time...
Term Term::vsubst(Term tm0, Vec<Subst> const& subs)
{
    uint subs_mask = 0;
    for (Subst const& sub : subs)
        subs_mask |= varMask(sub.x.var());

    // Phase 1 -- determine new lambda-variable indices:
    Vec<Term> lvars;
    Vec<bool> ignore;

  #if (__cplusplus > 201103L)
    auto renameLambs = [&](auto const& me, Term tm) -> uint {
        auto renameLambs = [&me](Term tm){ return me(me, tm); };
        // -- recursion with type 'function<uint(Term)>' has too much overhead;
        // pass ourself as first parameter instead.
  #else
    function<uint(Term)> renameLambs = [&](Term tm) -> uint {
  #endif
        ignore.push((tm->var_mask & subs_mask) == 0);
        if (ignore.last()) return tm->lambda_c;

        switch (tm.kind()){
        case Term::VAR:
            for (Subst const& sub : subs)
                if (sub.x == tm)
                    return sub.tm->lambda_c;
            return 0;
        case Term::CNST:
            return 0;
        case Term::COMB: {
            uint idx_f = renameLambs(tm.fun());
            uint idx_a = renameLambs(tm.arg());
            return max_(idx_f, idx_a); }
        case Term::ABS: {
            uint i = lvars.size();
            lvars.push();
            uint new_idx = renameLambs( tm.aterm());
            lvars[i] = Term(Tag_unsafe_var(), lambdaVar(new_idx), tm.avar().type());
            return new_idx+1; }
        default: assert(false); }
    };
  #if (__cplusplus > 201103L)
    renameLambs(renameLambs, tm0);
  #else
    renameLambs(tm0);
  #endif

    // Phase 2 -- perform variable substitutions:
    uint i = 0, j = 0;
  #if (__cplusplus > 201103L)
    auto doSubst = [&](auto&& me, Term tm) -> Term {
        auto doSubst = [&me](Term tm){ return me(me, tm); };
  #else
        function<Term(Term)> doSubst = [&](Term tm) -> Term {
  #endif
        if (ignore[i++]) return tm;

        switch (tm.kind()){
        case Term::VAR:
            for (Subst const& sub : subs)
                if (sub.x == tm) return sub.tm;
            return tm;
        case Term::CNST:
            return tm;
        case Term::COMB: {
            Term tm_f = doSubst(tm.fun());
            Term tm_a = doSubst(tm.arg());
            if (tm.fun() == tm_f && tm.arg() == tm_a) return tm;
            return tmComb(tm_f, tm_a); }
        case Term::ABS: {
            Term alpha = lvars[j++];  // -- first (depth-first) substitute the lambda-var (problematic to piggy-back on 'subs')
            Term tm_aterm = doSubst((tm.avar() == alpha) ? tm.aterm() : subst(tm.aterm(), tm.avar(), alpha));
            if (tm.avar() == alpha && tm.aterm() == tm_aterm) return tm;
            return Term(Tag_unsafe_abs(), alpha, tm_aterm, tm.type()); }
        default: assert(false); }
    };
  #if (__cplusplus > 201103L)
    return doSubst(doSubst, tm0);
  #else
    return doSubst(tm0);
  #endif
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
