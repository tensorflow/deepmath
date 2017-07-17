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
#include "Kernel.hh"
#include "ParserTypes.hh"
#include "Printing.hh"
  // -- 'ParserTypes.hh' for proof-logging (using 'RuleKind' enum)
  // -- 'Printing.hh' for debugging

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Local Helpers:


inline Type funType(Type ty_A, Type ty_B) {
    return Type(tcon_fun, mkList({ty_A, ty_B})); }


inline Type eqType(Type ty) {
    return funType(ty, funType(ty, type_bool)); }


inline Term makeEqTerm(Term tm1, Term tm2) {
    assert(tm1.type() == tm2.type());
    Term eq = tmCnst(cnst_eq, eqType(tm1.type()));
    return tmComb(tmComb(eq, tm1), tm2); }


// Return TRUE if 'tm' is on form: `((= x) y)`
inline bool isEqTerm(Term tm) {
    return tm.is_comb()
        && tm.fun().is_comb()
        && tm.fun().fun().is_cnst()
        && tm.fun().fun().cnst() == cnst_eq; }


static SSet<TVar> setOf_tvars(Type ty)
{
    if (ty.is_tvar())
        return SSet<TVar>(ty.tvar());

    SSet<TVar> acc;
    for (List<Type> it = ty.targs(); it; ++it)
        acc |= setOf_tvars(*it);
    return acc;
};


static SSet<TVar> setOf_tvars(Term tm)
{
    switch (tm.kind()){
    case Term::VAR : return setOf_tvars(tm.type());
    case Term::CNST: return setOf_tvars(tm.type());
    case Term::COMB: return setOf_tvars(tm.fun()) | setOf_tvars(tm.arg());
    case Term::ABS : return setOf_tvars(tm.avar()) | setOf_tvars(tm.aterm());
    default: assert(false); }
}


static bool hasFreeVars(Term tm)
{
    switch (tm.kind()){
    case Term::VAR : return !Term::isLambdaVar(tm.var());
    case Term::CNST: return false;
    case Term::COMB: return hasFreeVars(tm.fun()) || hasFreeVars(tm.arg());
    case Term::ABS : return hasFreeVars(tm.aterm());
    default: assert(false); }
}


// Is 'sub_ty' a specialization of 'base_ty'? Populates substitution map as a side-effect.
static bool typeMatch(Type base_ty, Type sub_ty, /*out*/Vec<TSubst>& subs)
{
    if (base_ty.is_tvar()){
        for (TSubst& s : subs)
            if (base_ty == s.a)
                return sub_ty == s.ty;      // -- substitutions must be consistent
        subs.push(TSubst(base_ty, sub_ty));
        return true;

    }else{
        if (sub_ty.is_tvar() || base_ty.tcon() != sub_ty.tcon())
            return false;

        List<Type> bs = base_ty.targs();
        for (List<Type> ss = sub_ty.targs(); ss; ++ss, ++bs){
            assert(bs);     // -- type-lists should have same length
            if (!typeMatch(*bs, *ss, subs))
                return false;
        }
        assert(!bs);
        return true;
    }
}


template<class T>
inline size_t size(List<T> list) {
    size_t sz = 0;
    for (List<T> it = list; it; ++it) sz++;
    return sz; }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Kernel state variables:


bool kernel_proof_logging = false;

static Vec<Ax> kernel_axioms;
    // In 'fusion.ml' it's called "the_axioms". Elements are populated from from 'kernel_New_Ax()'.

static Vec<Def> kernel_consts;
    // In 'fusion.ml' it's called "the_term_constants". Except for primitive constants '=', '@' and
    // internal '`&', elements are populated from 'kernel_New_Def()' and 'kernel_New_TDef'.

static Vec<TDef> kernel_typecons;
    // In 'fusion.ml' it's called "the_type_constants". Except for the primitive types 'bool' and
    // 'fun', all elements are constructed from 'kernel_New_TDef()'.


ZZ_Initializer(predefined_typecons_and_consts, 10) {    // -- prio 10 to go after initialization in 'Types.cc'
    kernel_typecons(+tcon_bool) = TDef{0, tcon_bool, Cnst(), Cnst(), Thm(), Thm()};
    kernel_typecons(+tcon_fun ) = TDef{2, tcon_fun , Cnst(), Cnst(), Thm(), Thm()};
    kernel_typecons(+tcon_ind ) = TDef{0, tcon_ind , Cnst(), Cnst(), Thm(), Thm()};

    kernel_consts(+cnst_eq  ) = Def{cnst_eq  , tmCnst(cnst_eq  , eqType(type_alpha))};  // -- built-in constants have no further definition (define them to themselves)
    kernel_consts(+cnst_hilb) = Def{cnst_hilb, tmCnst(cnst_hilb, funType(funType(type_alpha, type_bool), type_alpha))};
    kernel_consts(+cnst_iand) = Def{cnst_iand, tmCnst(cnst_iand, type_booleq)};
}


Ax   getAxiom(Axiom ax) { assert(+ax < kernel_axioms  .size()); return kernel_axioms  [+ax]; }
Def  getDef  (Cnst  c ) { assert(+c  < kernel_consts  .size()); return kernel_consts  [+c ]; }
TDef getTDef (TCon  tc) { assert(+tc < kernel_typecons.size()); return kernel_typecons[+tc]; }


template<class T>
Vec<T> nonNull(Vec<T> const& ts) {
    Vec<T> ret;
    for (auto&& t : ts) if (t) ret.push(t);
    return ret; }

Vec<Ax>   kernelAxioms  () { return nonNull(kernel_axioms  ); }
Vec<Def>  kernelConsts  () { return nonNull(kernel_consts  ); }
Vec<TDef> kernelTypecons() { return nonNull(kernel_typecons); }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


// Profiling:
ZZ_PTimer_Add(kernel_New_Def);
ZZ_PTimer_Add(kernel_REFL);
ZZ_PTimer_Add(kernel_MK_COMB);
ZZ_PTimer_Add(kernel_EQ_MP);
ZZ_PTimer_Add(kernel_ASSUME);
ZZ_PTimer_Add(kernel_DEDUCT);
ZZ_PTimer_Add(kernel_BETA);
ZZ_PTimer_Add(kernel_INST);
ZZ_PTimer_Add(kernel_TRANS);
ZZ_PTimer_Add(kernel_ABS);
ZZ_PTimer_Add(kernel_INST_T);
ZZ_PTimer_Add(kernel_New_Ax);
ZZ_PTimer_Add(kernel_New_TDef);
ZZ_PTimer_Add(kernel_Inst_Cnst);


//    t
// --------
// |- t = t
//
Thm kernel_REFL(Term tm) {
    ZZ_PTimer_Scope(kernel_REFL);
    return Thm(makeEqTerm(tm, tm), logStep(rule_REFL, tm)); }


//    (\x. t) x
// ----------------  [if types match between bound and unbound `x`]
// |- (\x. t) x = t
//
Thm kernel_BETA(Term tm)
{
    ZZ_PTimer_Scope(kernel_BETA);
    // NOTE! We actually implement a stronger rule: `(\y. t) x = t[y := x]` for unique lambda-var `y`
    // This means that a proof in this kernel might not check with Cezary's kernel (but it can be easily fixed)
    assert(tm.is_comb());
    Term lam = tm.fun(); assert(lam.is_abs());
    Term x   = tm.arg(); assert(x.is_var()); assert(x.type() == lam.avar().type());
    Term t   = lam.betaRed(x);
    return Thm(makeEqTerm(tm, t), logStep(rule_BETA, tm));
}


// t : bool
// --------
//  t |- t
//
Thm kernel_ASSUME(Term tm)
{
    ZZ_PTimer_Scope(kernel_ASSUME);
    assert(tm.type() == type_bool);
    return Thm(SSet<Term>(tm), tm, logStep(rule_ASSUME, tm));
}


//  h |- l = r     x
// ------------------  [if variable `x` is not free in `h`]
//  h |- \x.l = \x.r
//
Thm kernel_ABS(Thm th, Term x)
{
    ZZ_PTimer_Scope(kernel_ABS);
    assert(x.is_var());
    for (List<Term> it = th.hyps(); it; ++it)
        assert(!(*it).hasVar(x));

    Term l = th.concl().fun().arg();
    Term r = th.concl().arg();
    return Thm(th.hyps(), makeEqTerm(tmAbs(x, l), tmAbs(x, r)), logStep(rule_ABS, th, x));
}


// h1 |- l = m    h2 |- m' = r
// ---------------------------  [if `m` alpha-equal to `m'`]
//       h1, h2 |- l = r
//
Thm kernel_TRANS(Thm th1, Thm th2)
{
    ZZ_PTimer_Scope(kernel_TRANS);
    Term eq1 = th1.concl(); assert(isEqTerm(eq1));
    Term eq2 = th2.concl(); assert(isEqTerm(eq2));
    Term l  = eq1.fun().arg();
    Term m  = eq1.arg();
    Term m_ = eq2.fun().arg();
    Term r  = eq2.arg();
    assert(m == m_);
    return Thm(th1.hyps() | th2.hyps(), makeEqTerm(l, r), logStep(rule_TRANS, th1, th2));
}


// h1 |- f = g    h2 |- a = b
// --------------------------  [if types match]
//     h1, h2 |- f a = g b
//
Thm kernel_MK_COMB(Thm funs_eq, Thm args_eq)        // -- congruence
{
    ZZ_PTimer_Scope(kernel_MK_COMB);
    Term fs = funs_eq.concl(); assert(isEqTerm(fs));
    Term as = args_eq.concl(); assert(isEqTerm(as));

    Term f = fs.fun().arg();  // -- extract `f` from `((= f) g)`
    Term a = as.fun().arg();
    Term g = fs.arg();
    Term b = as.arg();
    Type ty = f.type();
    assert(ty.is_tapp()); assert(ty.tcon() == tcon_fun);    // -- `f` must be of type A->B
    assert(ty.targs()[0] == a.type());  // -- `a` must match input type of `f`
    assert(g.type() == ty);             // }- redundant tests, missing from Cezary's code
    assert(a.type() == b.type());       // }  (presumably you cannot create `|- x = y` with mismatching types for `x` and `y`)

    return Thm(funs_eq.hyps() | args_eq.hyps(), makeEqTerm(tmComb(f, a), tmComb(g, b)), logStep(rule_MK_COMB, funs_eq, args_eq));
}


// h1 |- f = g    h2 |- f'
// -----------------------  [if `f` is alpha-equal to `f'`]
//      h1, h2 |- g
//
Thm kernel_EQ_MP(Thm props_eq, Thm prop)        // -- equality modus ponens
{
    ZZ_PTimer_Scope(kernel_EQ_MP);
    Term f_ = prop.concl();
    Term eq = props_eq.concl(); assert(isEqTerm(eq));
    Term f = eq.fun().arg();    // -- extract `f` from `((= f) g)`
    Term g = eq.arg();
    assert(f == f_);            // -- alpha-equality is just equality due to term normalization

    return Thm(props_eq.hyps() | prop.hyps(), g, logStep(rule_EQ_MP, props_eq, prop));
}


//      h1 |- c1    h2 |- c2
// ---------------------------------
//  (h1\{c2}), (h2\{c1}) |- c1 = c2
//
Thm kernel_DEDUCT(Thm th1, Thm th2)
{
    ZZ_PTimer_Scope(kernel_DEDUCT);
    Term c1 = th1.concl();
    Term c2 = th2.concl();
    return Thm(th1.hyps().exclude(c2) | th2.hyps().exclude(c1), makeEqTerm(c1, c2), logStep(rule_DEDUCT, th1,th2));
}


//       h |- p       (x1, t1), ..., (xn, tn)
// ------------------------------------------------
// h[x1:=t1, ..., xn:=tn] |- p[x1:=t1, ..., xn:=tn]
//
Thm kernel_INST(Thm th, Vec<Subst>& subs)
{
    ZZ_PTimer_Scope(kernel_INST);
    auto applySubst = [&](Term tm) { return tm.varSubst(subs); };
    return Thm(th.hyps().map(applySubst), applySubst(th.concl()), logStep(rule_INST, th, mkList_substs(subs)));
}


//    h |- p       (a1, T1), ..., (an, Tn)
// --------------------------------------------
// h[a1:=T1,...,an:=Tn] |- p[a1:=T1,...,an:=Tn]
//
Thm kernel_INST_T(Thm th, Vec<TSubst>& subs)
{
    ZZ_PTimer_Scope(kernel_INST_T);
    auto applySubst = [&](Term tm) { return tm.typeSubst(subs); };
    return Thm(th.hyps().map(applySubst), applySubst(th.concl()), logStep(rule_INST_T, th, mkList_tsubsts(subs)));
}


// t : bool
// --------
//   |- t
//
Thm kernel_New_Ax(Axiom name, Term tm)
{
    ZZ_PTimer_Scope(kernel_New_Ax);
    assert(tm.type() == type_bool);
    kernel_axioms(+name) = Ax{name, Thm(tm, logStep(rule_New_Ax, name, tm))};
    return kernel_axioms[+name].th;
}


//   c  t
// --------  [if `t` has no free variables and all type-variables are in its type]
// |- c = t
//
Thm kernel_New_Def(Cnst name, Term tm)     // -- new basic definition
{
    ZZ_PTimer_Scope(kernel_New_Def);
    assert(!hasFreeVars(tm));
    assert(setOf_tvars(tm).subsetOf(setOf_tvars(tm.type())));
    Thm ret = Thm(makeEqTerm(tmCnst(name, tm.type()), tm), logStep(rule_New_Def, name, tm));
    kernel_consts(+name) = Def{name, tm, ret};
    return ret;
}


//    tc_name   abs_name   rep_name   |- P t
// ---------------------------------------------  [`tc`, `abs` and `rep` are previously undefined]
// |- abs(rep a) = a   |- P r = (rep(abs r) = r)
//
// where: `tc`=new type-constructor, `abs`=abstraction, `rep`=representation
// and:   `P` = property, `t` witness (`|- P t` proof of existence)
//
// types are:
//     P   : old_type -> bool
//     abs : old_type -> new_type
//     rep : new_type -> old_type
//
// NOTE: The arity of the new type-constructor is determined by the number of free
// type-variables in `P`.
//
// From 'fusion.ml': This function now involves no logical constants beyond equality.
//
Thm kernel_New_TDef(TCon tc_name, Cnst abs_name, Cnst rep_name, Thm th_wit)
{
    ZZ_PTimer_Scope(kernel_New_TDef);
    // NOTE! 'tc_name' is the name of the new type-constructor. If it has no type-argument, the
    // same name will often be used for the type constructed from it (e.g. "bool", but not "list"
    // which takes an argument "list(A)" and specialize it into "list(num)" or "list(bool)".)
    assert(th_wit.hyps().empty());
    assert(th_wit.concl().is_comb());
    Term P = th_wit.concl().fun();
    Term t = th_wit.concl().arg();
    assert(!hasFreeVars(P));

    SSet<TVar> P_tvars = setOf_tvars(P);
    uint arity = P_tvars.size();
    assert(!kernel_typecons(+tc_name)); // -- type-constructor must have a new name
    SSet<Type> tc_args = P_tvars.map([](TVar v) { return Type(v); });
    Type ty_new = Type(tc_name, tc_args);
    Type ty_old = t.type();
    Type ty_abs = funType(ty_old, ty_new);
    Type ty_rep = funType(ty_new, ty_old);

    assert(!kernel_consts(+abs_name));       // }- map functions must have new names (as constants)
    assert(!kernel_consts(+rep_name));       // }
    Term abs = tmCnst(abs_name, ty_abs);
    Term rep = tmCnst(rep_name, ty_rep);
    kernel_consts(+abs_name) = Def{abs_name, abs};
    kernel_consts(+rep_name) = Def{rep_name, rep};

    Term a = tmVar(Var("a"), ty_new);
    Term r = tmVar(Var("r"), ty_old);

    Term ret1 =                          makeEqTerm(tmComb(abs, tmComb(rep, a)), a);
    Term ret2 = makeEqTerm(tmComb(P, r), makeEqTerm(tmComb(rep, tmComb(abs, r)), r));

    Term iand = tmCnst(cnst_iand, type_booleq);
    Thm  ret = Thm(tmComb(tmComb(iand, ret1), ret2), logStep(rule_New_TDef, tc_name, abs_name, rep_name, th_wit));
    kernel_typecons(+tc_name) = TDef{arity, tc_name, abs_name, rep_name, th_wit, ret};
    return ret;
}


Thm kernel_Extract1(Thm th)
{
    assert(th.hyps().empty());
    Term tm = th.concl();
    assert(tm.is_comb());
    assert(tm.fun().is_comb());
    assert(tm.fun().fun().cnst() == cnst_iand);
    return Thm(tm.fun().arg(), logStep(rule_TDef_Ex1, th));
}


Thm kernel_Extract2(Thm th)
{
    assert(th.hyps().empty());
    Term tm = th.concl();
    assert(tm.is_comb());
    assert(tm.fun().is_comb());
    assert(tm.fun().fun().cnst() == cnst_iand);
    return Thm(tm.arg(), logStep(rule_TDef_Ex2, th));
}


// c     (a1, T1), ..., (an, Tn)
// -----------------------------  [`c` is a constant-constructor introduced by 'kernel_New_Def()']
//      c[a1:=T1,...,an:=Tn]
//
// Note: special constant-constructors 'cnst_eq' and 'cnst_hilb' are predefined.
//
Term kernel_Inst_Cnst(Cnst c, Vec<TSubst>& subs)
{
    ZZ_PTimer_Scope(kernel_Inst_Cnst);
    assert(+c < kernel_consts.size());
    return tmCnst(c, kernel_consts[+c].def.type()).typeSubst(subs);
}


// tc     T1, ..., Tn
// ------------------  [`tc` is a type-constructor introduced by 'kernel_New_TDef()'
//   T<T1, ..., Tn>
//
// Note: special type-constructors 'tcon_bool', 'tcon_fun' and 'tcon_ind' are predefined.
//
Type kernel_Inst_Type(TCon tc, List<Type> ts)
{
    assert(+tc < kernel_typecons.size());
    assert(kernel_typecons[+tc].arity == size(ts));
    return Type(tc, ts);
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Safe helpers: (respects kernel contract)


Type typeOf(Term tm)               { return tm.type();               }
Type mkTVar(TVar v)                { return Type(v);                 }
Type mkTApp(TCon c, List<Type> ts) { return kernel_Inst_Type(c, ts); }
Term mkVar (Var  x, Type ty)       { return tmVar (x, ty);           }
Term mkComb(Term f, Term tm)       { return tmComb(f, tm);           }
Term mkAbs (Term x, Term tm)       { return tmAbs (x, tm);           }

Term mkCnst(Cnst c, Type ty) {
    Vec<TSubst> subs;
    bool ok = typeMatch(getDef(c).def.type(), ty, subs); assert(ok);
    Term ret = kernel_Inst_Cnst(c, subs); assert(ret.type() == ty);
    return ret; }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
